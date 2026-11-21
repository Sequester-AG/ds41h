#include "http/http_server.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <ctime>
#include <stdexcept>

#ifdef FE_HTTP_TLS
#include <openssl/err.h>
#include <openssl/ssl.h>
#endif

struct ssl_st;  // OpenSSL's SSL

namespace fe::http {

namespace {

inline char lower(char c) { return (c >= 'A' && c <= 'Z') ? char(c + 32) : c; }

std::string to_lower(std::string_view s) {
  std::string r(s);
  for (auto& c : r) c = lower(c);
  return r;
}

bool ieq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (lower(a[i]) != lower(b[i])) return false;
  return true;
}

std::string_view trim_ows(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
  return s;
}

int hexval(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// urllib.parse.unquote (UTF-8, errors="replace"): invalid escapes are kept verbatim.
std::string percent_decode(std::string_view s, bool plus_space) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    char c = s[i];
    if (c == '%' && i + 2 < s.size() && hexval(s[i + 1]) >= 0 && hexval(s[i + 2]) >= 0) {
      out.push_back(char(hexval(s[i + 1]) * 16 + hexval(s[i + 2])));
      i += 2;
    } else if (plus_space && c == '+') {
      out.push_back(' ');
    } else {
      out.push_back(c);
    }
  }
  return out;
}

void set_nonblock(int fd) {
  int fl = fcntl(fd, F_GETFL, 0);
  if (fl >= 0) fcntl(fd, F_SETFL, fl | O_NONBLOCK);
}

std::string fmt_http_date(time_t t) {
  static const char* days[] = {"Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"};
  static const char* mons[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  struct tm g;
  gmtime_r(&t, &g);
  char buf[64];
  snprintf(buf, sizeof buf, "%s, %02d %s %04d %02d:%02d:%02d GMT", days[g.tm_wday], g.tm_mday,
           mons[g.tm_mon], g.tm_year + 1900, g.tm_hour, g.tm_min, g.tm_sec);
  return buf;
}

#ifdef FE_HTTP_TLS
// The first (root-cause) error of this thread's OpenSSL queue, without its data part (which can hold a
// file path), then clears the queue.
std::string ssl_err() {
  unsigned long e = ERR_get_error();
  ERR_clear_error();
  if (!e) return "unknown error";
  char b[256];
  ERR_error_string_n(e, b, sizeof b);
  return b;
}

// Encrypted private keys fail to load instead of OpenSSL prompting for a passphrase on the tty.
int no_password_cb(char*, int, int, void*) { return 0; }

// ALPN: we only speak HTTP/1.1. A client that offers it gets it; anything else gets no ALPN.
int alpn_select_cb(SSL*, const unsigned char** out, unsigned char* outlen, const unsigned char* in,
                   unsigned int inlen, void*) {
  static const unsigned char kHttp11[] = {8, 'h', 't', 't', 'p', '/', '1', '.', '1'};
  unsigned char* sel = nullptr;
  if (SSL_select_next_proto(&sel, outlen, kHttp11, sizeof kHttp11, in, inlen) == OPENSSL_NPN_NEGOTIATED) {
    *out = sel;
    return SSL_TLSEXT_ERR_OK;
  }
  return SSL_TLSEXT_ERR_NOACK;
}

// CPython 3.12's PY_SSL_DEFAULT_CIPHER_STRING (what uvicorn's SSLContext uses for TLS <= 1.2).
const char* const kPythonCiphers =
    "@SECLEVEL=2:ECDH+AESGCM:ECDH+CHACHA20:ECDH+AES:DHE+AES:!aNULL:!eNULL:!aDSS:!SHA1:!AESCCM";

// Builds the server SSL_CTX like uvicorn's create_ssl_context (+ TLS >= 1.2, ALPN http/1.1, no
// renegotiation). Throws std::runtime_error; messages never contain the key path or any file content.
SSL_CTX* make_tls_ctx(const ServerOptions& o) {
  if (o.tls_certfile.empty()) throw std::runtime_error("TLS: --ssl-certfile is required");
  ERR_clear_error();
  SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
  if (!ctx) throw std::runtime_error("TLS: SSL_CTX_new failed: " + ssl_err());
  auto fail = [&](const std::string& m) {
    SSL_CTX_free(ctx);
    throw std::runtime_error(m);
  };
  SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
  SSL_CTX_set_options(ctx, SSL_OP_NO_COMPRESSION | SSL_OP_CIPHER_SERVER_PREFERENCE | SSL_OP_NO_RENEGOTIATION);
  SSL_CTX_set_mode(ctx, SSL_MODE_ENABLE_PARTIAL_WRITE | SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER |
                            SSL_MODE_RELEASE_BUFFERS);
  SSL_CTX_set_default_passwd_cb(ctx, no_password_cb);
  static const unsigned char sid[] = "fe_http";
  SSL_CTX_set_session_id_context(ctx, sid, sizeof sid - 1);
  const char* ciphers = o.tls_ciphers.empty() ? kPythonCiphers : o.tls_ciphers.c_str();
  if (SSL_CTX_set_cipher_list(ctx, ciphers) != 1) fail("TLS: invalid --ssl-ciphers: " + ssl_err());
  if (SSL_CTX_use_certificate_chain_file(ctx, o.tls_certfile.c_str()) != 1)
    fail("TLS: cannot load the certificate chain from " + o.tls_certfile + ": " + ssl_err());
  const std::string& kf = o.tls_keyfile.empty() ? o.tls_certfile : o.tls_keyfile;
  if (SSL_CTX_use_PrivateKey_file(ctx, kf.c_str(), SSL_FILETYPE_PEM) != 1)
    fail("TLS: cannot load the private key (--ssl-keyfile): " + ssl_err());
  if (SSL_CTX_check_private_key(ctx) != 1) fail("TLS: the private key does not match the certificate: " + ssl_err());
  if (!o.tls_ca_certs.empty() && SSL_CTX_load_verify_locations(ctx, o.tls_ca_certs.c_str(), nullptr) != 1)
    fail("TLS: cannot load CA certificates from " + o.tls_ca_certs + ": " + ssl_err());
  int vm = SSL_VERIFY_NONE;
  if (o.tls_cert_reqs == 1)
    vm = SSL_VERIFY_PEER;
  else if (o.tls_cert_reqs == 2)
    vm = SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT;
  else if (o.tls_cert_reqs != 0)
    fail("TLS: --ssl-cert-reqs must be 0 (none), 1 (optional) or 2 (required)");
  SSL_CTX_set_verify(ctx, vm, nullptr);
  SSL_CTX_set_alpn_select_cb(ctx, alpn_select_cb, nullptr);
  ERR_clear_error();
  return ctx;
}
#endif

}  // namespace

// ---------------------------------------------------------------------------------------
// Request helpers

const std::string* Request::header(std::string_view lname) const {
  for (auto& h : headers)
    if (h.first == lname) return &h.second;
  return nullptr;
}

std::optional<std::string> Request::query_param(std::string_view name) const {
  std::string_view q = query;
  while (!q.empty()) {
    size_t amp = q.find('&');
    std::string_view kv = q.substr(0, amp);
    q = amp == std::string_view::npos ? std::string_view() : q.substr(amp + 1);
    if (kv.empty()) continue;
    size_t eq = kv.find('=');
    std::string k = percent_decode(kv.substr(0, eq), true);
    if (k == name) return percent_decode(eq == std::string_view::npos ? "" : kv.substr(eq + 1), true);
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------------------
// Connection

class Conn : public std::enable_shared_from_this<Conn> {
 public:
  Server* srv = nullptr;
  int fd = -1;
  uint64_t id = 0;
  std::string client;

  // ---- IO-thread state ----
  enum class St { Handshake, Headers, Body, Chunked, Dispatched, Closed };  // Handshake: TLS only
  St st = St::Headers;
  std::string in;
  std::shared_ptr<Request> req;
  size_t body_left = 0;
  enum class Ch { Size, Data, DataEnd, Trailer };
  Ch ch = Ch::Size;
  size_t chunk_left = 0;
  uint32_t interest = 0;
  bool had_request = false;
  std::chrono::steady_clock::time_point idle_since = std::chrono::steady_clock::now();
  std::shared_ptr<Response> resp;  // current in-flight response

  // ---- shared output state (under mu) ----
  std::mutex mu;
  std::string out;
  size_t out_off = 0;
  bool resp_done = false;      // current response fully queued
  bool want_out = false;       // IO thread owns flushing (EPOLLOUT armed or scheduled)
  bool closed = false;         // fd closed / connection dead
  bool write_error = false;
  bool congested = false;

  // ---- TLS (ssl != nullptr). Every SSL_* call is made under mu: the IO thread runs the handshake and
  // SSL_read, any thread SSL_write (flush_locked), close_conn SSL_shutdown + SSL_free. SSL reads from the
  // socket but WRITES INTO A MEMORY BIO: its records (application data, handshake, KeyUpdate replies,
  // tickets, alerts, close_notify) are moved to tls_out in production order and send(2)-ed from there by
  // the same flush path as plain bytes. So no SSL_* call ever waits for a writable socket: OpenSSL cannot
  // interleave a post-handshake message written from inside SSL_read (the reply to a TLS 1.3 KeyUpdate)
  // with an application record still pending in its own write buffer (SSL_R_BAD_LENGTH /
  // BAD_WRITE_RETRY, seen in test_http_tls), and a peer reset cannot raise SIGPIPE (MSG_NOSIGNAL).
  ssl_st* ssl = nullptr;
  std::string tls_out;             // under mu: ciphertext not yet sent
  size_t tls_out_off = 0;
  bool tls_hs_done = false;        // under mu
  bool tls_fatal = false;          // under mu: fatal TLS / socket error -> no close_notify
  bool tls_wr_want_read = false;   // under mu: SSL_write needs input first (renegotiation-like; not TLS 1.3)
  std::chrono::steady_clock::time_point hs_deadline;  // IO thread: end of the handshake allowance

  // plaintext not yet encrypted + ciphertext not yet sent (plain HTTP: tls_out is always empty)
  size_t pending_locked() const { return out.size() - out_off + (tls_out.size() - tls_out_off); }

#ifdef FE_HTTP_TLS
  // Moves what OpenSSL wrote into the memory BIO to tls_out.
  void tls_collect_locked() {
    BIO* w = SSL_get_wbio(ssl);
    char* p = nullptr;
    long n = BIO_get_mem_data(w, &p);
    if (n > 0) {
      tls_out.append(p, size_t(n));
      (void)BIO_reset(w);
    }
  }

  // send(2)s tls_out; leaves the rest on EAGAIN; false (write_error) on a hard socket error.
  bool tls_send_locked() {
    while (tls_out_off < tls_out.size()) {
      ssize_t n = ::send(fd, tls_out.data() + tls_out_off, tls_out.size() - tls_out_off, MSG_NOSIGNAL);
      if (n > 0) {
        tls_out_off += size_t(n);
      } else if (n < 0 && errno == EINTR) {
        continue;
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        break;
      } else {
        write_error = true;
        tls_fatal = true;
        return false;
      }
    }
    if (tls_out_off == tls_out.size()) {
      tls_out.clear();
      tls_out_off = 0;
    } else if (tls_out_off > (1 << 20) && tls_out_off * 2 > tls_out.size()) {
      tls_out.erase(0, tls_out_off);
      tls_out_off = 0;
    }
    return true;
  }

  // After an IO-thread SSL call (handshake, SSL_read): ship what it wrote; leftovers go to EPOLLOUT.
  void tls_after_io_locked() {
    tls_collect_locked();
    if (tls_out_off < tls_out.size()) {
      tls_send_locked();
      if (tls_out_off < tls_out.size() || write_error) want_out = true;
    }
  }

  // flush_locked() for TLS. Plaintext is encrypted (SSL_write, partial writes, <= ~64 KB of ciphertext
  // at a time) only while the socket takes the previous records, so the plaintext in `out` stays the
  // backpressure measure and tls_out stays small.
  bool tls_flush_locked() {
    for (;;) {
      if (!tls_send_locked()) return false;
      if (tls_out_off < tls_out.size() || out_off >= out.size() || tls_wr_want_read) break;
      while (out_off < out.size() && tls_out.size() - tls_out_off < (64u << 10)) {
        size_t len = std::min(out.size() - out_off, size_t(1) << 20);
        ERR_clear_error();
        int n = SSL_write(ssl, out.data() + out_off, int(len));
        int e = n > 0 ? SSL_ERROR_NONE : SSL_get_error(ssl, n);
        ERR_clear_error();
        tls_collect_locked();
        if (n > 0) {
          out_off += size_t(n);
          continue;
        }
        if (e == SSL_ERROR_WANT_READ) {  // resumes after the next read (on_readable)
          tls_wr_want_read = true;
          break;
        }
        if (e == SSL_ERROR_SSL || e == SSL_ERROR_SYSCALL) tls_fatal = true;
        write_error = true;  // incl. WANT_WRITE, impossible with a memory BIO (allocation failure)
        return false;
      }
    }
    if (out_off == out.size()) {
      out.clear();
      out_off = 0;
    } else if (out_off > (1 << 20) && out_off * 2 > out.size()) {
      out.erase(0, out_off);
      out_off = 0;
    }
    return true;
  }
#endif

  // Writes as much as possible; returns false on a hard error.
  bool flush_locked() {
#ifdef FE_HTTP_TLS
    if (ssl) return tls_flush_locked();
#endif
    while (out_off < out.size()) {
      ssize_t n = ::send(fd, out.data() + out_off, out.size() - out_off, MSG_NOSIGNAL);
      if (n > 0) {
        out_off += size_t(n);
      } else if (n < 0 && errno == EINTR) {
        continue;
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        break;
      } else {
        write_error = true;
        return false;
      }
    }
    if (out_off == out.size()) {
      out.clear();
      out_off = 0;
    } else if (out_off > (1 << 20) && out_off * 2 > out.size()) {
      out.erase(0, out_off);
      out_off = 0;
    }
    return true;
  }

  // Queue bytes from any thread. Returns false if the connection is gone.
  bool queue(std::string_view data, bool last) {
    bool schedule = false;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (closed || write_error) return false;
      out.append(data.data(), data.size());
      if (last) resp_done = true;
      if (!want_out) {
        flush_locked();
        if (write_error || pending_locked() > 0 || last) {
          want_out = true;
          schedule = true;
        }
      }
      if (pending_locked() > srv->opts_.high_water) congested = true;
    }
    if (schedule) srv->schedule_flush(shared_from_this());
    return true;
  }
};

// ---------------------------------------------------------------------------------------
// Response

Response::Response(std::shared_ptr<Conn> c, bool head, bool keep_alive)
    : conn_(c), head_method_(head), keep_alive_(keep_alive) {}

Response::~Response() = default;

std::string Response::build_head(int status, const HeaderList& headers, bool has_length,
                                 size_t length, bool* chunked_out) {
  if (!access_line_.empty()) {
    static std::mutex log_mu;
    std::lock_guard<std::mutex> lk(log_mu);
    fprintf(stderr, "INFO:     %s %d %s\n", access_line_.c_str(), status, std::string(status_phrase(status)).c_str());
    fflush(stderr);
  }
  std::string h;
  h.reserve(256);
  h += "HTTP/1.1 ";
  h += std::to_string(status);
  h += ' ';
  h += status_phrase(status);
  h += "\r\n";
  auto c = conn_.lock();
  if (c) h += c->srv->default_headers();
  bool chunked = false, have_cl = false, conn_close = false;
  for (auto& [n0, v] : headers) {
    std::string n = to_lower(n0);
    if (n == "content-length") have_cl = true;
    if (n == "transfer-encoding" && ieq(v, "chunked")) chunked = true;
    if (n == "connection" && v.find("close") != std::string::npos) conn_close = true;
    h += n;
    h += ": ";
    h += v;
    h += "\r\n";
  }
  if (has_length && !have_cl) {
    h += "content-length: ";
    h += std::to_string(length);
    h += "\r\n";
    have_cl = true;
  }
  if (conn_close) keep_alive_ = false;
  if (!keep_alive_ && !conn_close) h += "connection: close\r\n";
  if (!have_cl && !chunked && !head_method_ && status != 204 && status != 304) {
    chunked = true;
    h += "transfer-encoding: chunked\r\n";
  }
  h += "\r\n";
  *chunked_out = chunked;
  return h;
}

void Response::send(int status, HeaderList headers, std::string body) {
  if (started_.exchange(true)) return;
  auto c = conn_.lock();
  bool chunked = false;
  std::string data = build_head(status, headers, true, body.size(), &chunked);
  if (!head_method_) data += body;
  finished_.store(true, std::memory_order_release);
  if (c) c->queue(data, true);
}

void Response::start(int status, HeaderList headers) {
  if (started_.exchange(true)) return;
  auto c = conn_.lock();
  std::string data = build_head(status, headers, false, 0, &chunked_);
  if (c) c->queue(data, false);
}

bool Response::write(std::string_view data) {
  if (!started() || finished()) return false;
  auto c = conn_.lock();
  if (!c || disconnected()) return false;
  if (data.empty()) return true;
  if (head_method_) return true;
  if (chunked_) {
    char hdr[32];
    int n = snprintf(hdr, sizeof hdr, "%zx\r\n", data.size());
    std::string piece;
    piece.reserve(size_t(n) + data.size() + 2);
    piece.append(hdr, size_t(n));
    piece.append(data.data(), data.size());
    piece += "\r\n";
    return c->queue(piece, false);
  }
  return c->queue(data, false);
}

void Response::end() {
  if (!started() || finished_.exchange(true)) return;
  auto c = conn_.lock();
  if (!c) return;
  c->queue(chunked_ && !head_method_ ? std::string_view("0\r\n\r\n") : std::string_view(), true);
}

size_t Response::pending_bytes() const {
  auto c = conn_.lock();
  if (!c) return 0;
  std::lock_guard<std::mutex> lk(c->mu);
  return c->pending_locked();
}

bool Response::congested() const {
  auto c = conn_.lock();
  if (!c) return false;
  std::lock_guard<std::mutex> lk(c->mu);
  return c->congested;
}

void Response::on_disconnect(std::function<void()> cb) {
  bool fire_now = false;
  {
    std::lock_guard<std::mutex> lk(cb_mu_);
    if (disconnected() && !disconnect_fired_) {
      disconnect_fired_ = true;
      fire_now = true;
    } else {
      on_disconnect_ = std::move(cb);
    }
  }
  if (fire_now && cb) {
    auto c = conn_.lock();
    if (c)
      c->srv->pool().post(std::move(cb));
    else
      cb();
  }
}

void Response::on_writable(std::function<void()> cb) {
  std::lock_guard<std::mutex> lk(cb_mu_);
  on_writable_ = std::move(cb);
}

void Response::fire_disconnect() {
  disconnected_.store(true, std::memory_order_release);
  std::function<void()> cb, wake;
  {
    std::lock_guard<std::mutex> lk(cb_mu_);
    // A writer parked on congested() is woken once: it sees disconnected() and can release what it holds.
    wake = std::move(on_writable_);
    on_writable_ = nullptr;
    if (!disconnect_fired_ && on_disconnect_) {  // else: fired later in on_disconnect()
      disconnect_fired_ = true;
      cb = std::move(on_disconnect_);
    }
  }
  auto c = conn_.lock();
  for (auto* f : {&cb, &wake}) {
    if (!*f) continue;
    if (c)
      c->srv->pool().post(std::move(*f));
    else
      (*f)();
  }
}

void Response::fire_writable() {
  std::function<void()> cb;
  {
    std::lock_guard<std::mutex> lk(cb_mu_);
    cb = on_writable_;
  }
  if (!cb) return;
  auto c = conn_.lock();
  if (c) c->srv->pool().post(std::move(cb));
}

// ---------------------------------------------------------------------------------------
// WorkerPool

WorkerPool::WorkerPool(int n) {
  for (int i = 0; i < n; ++i) threads_.emplace_back([this] { run(); });
}
WorkerPool::~WorkerPool() { stop(); }
void WorkerPool::post(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) return;
    q_.push_back(std::move(fn));
  }
  cv_.notify_one();
}
size_t WorkerPool::queued() const {
  std::lock_guard<std::mutex> lk(mu_);
  return q_.size();
}
void WorkerPool::stop() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (stop_) {
      if (threads_.empty()) return;
    }
    stop_ = true;
  }
  cv_.notify_all();
  for (auto& t : threads_)
    if (t.joinable() && t.get_id() != std::this_thread::get_id()) t.join();
  threads_.clear();
}
void WorkerPool::run() {
  for (;;) {
    std::function<void()> fn;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
      if (q_.empty()) return;  // stop_ and drained
      fn = std::move(q_.front());
      q_.pop_front();
    }
    try {
      fn();
    } catch (const std::exception& e) {
      fprintf(stderr, "[fe_http] worker task threw: %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[fe_http] worker task threw\n");
    }
  }
}

// ---------------------------------------------------------------------------------------
// Server

Server::Server(ServerOptions opts, Handler handler) : opts_(std::move(opts)), handler_(std::move(handler)) {}

Server::~Server() {
  stop(0.0);
#ifdef FE_HTTP_TLS
  if (tls_ctx_) SSL_CTX_free(tls_ctx_);
  tls_ctx_ = nullptr;
#endif
}

bool Server::tls_available() {
#ifdef FE_HTTP_TLS
  return true;
#else
  return false;
#endif
}

TlsStats Server::tls_stats() const {
  TlsStats s;
  s.handshakes_ok = tls_ok_.load();
  s.handshake_failures = tls_fail_.load();
  s.handshake_timeouts = tls_timeout_.load();
  return s;
}

void Server::check_tls(const ServerOptions& o) {
#ifdef FE_HTTP_TLS
  SSL_CTX_free(make_tls_ctx(o));
#else
  (void)o;
  throw std::runtime_error("TLS requested (--ssl-certfile) but this frontend_cxx build has no TLS "
                           "(OpenSSL was not found at build time)");
#endif
}

void Server::tls_init() {
#ifdef FE_HTTP_TLS
  tls_ctx_ = make_tls_ctx(opts_);
#else
  check_tls(opts_);  // throws
#endif
}

std::string Server::default_headers() {
  std::string s;
  if (opts_.date_header) {
    time_t now = time(nullptr);
    std::lock_guard<std::mutex> lk(date_mu_);
    if (now != date_sec_) {
      date_sec_ = now;
      date_str_ = fmt_http_date(now);
    }
    s += "date: ";
    s += date_str_;
    s += "\r\n";
  }
  if (opts_.server_header) s += "server: uvicorn\r\n";
  return s;
}

void Server::start() {
  if (!opts_.tls_keyfile.empty() && opts_.tls_certfile.empty())
    throw std::runtime_error("TLS: --ssl-keyfile needs --ssl-certfile");
  if (!opts_.tls_certfile.empty() && !tls_ctx_) tls_init();
  if (opts_.listen_fd >= 0) {
    listen_fd_ = opts_.listen_fd;
  } else {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(uint16_t(opts_.port));
    if (inet_pton(AF_INET, opts_.host.c_str(), &a.sin_addr) != 1) a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) != 0)
      throw std::runtime_error(std::string("bind() failed: ") + strerror(errno));
    if (::listen(listen_fd_, 2048) != 0) throw std::runtime_error("listen() failed");
  }
  // The inherited socket is already bound (Python's create_server_socket) but may not be
  // listening yet: uvicorn's serve(sockets=...) relies on asyncio, which calls listen().
  ::listen(listen_fd_, 2048);
  set_nonblock(listen_fd_);
  {
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    if (getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&ss), &sl) == 0) {
      if (ss.ss_family == AF_INET)
        bound_port_ = ntohs(reinterpret_cast<sockaddr_in*>(&ss)->sin_port);
      else if (ss.ss_family == AF_INET6)
        bound_port_ = ntohs(reinterpret_cast<sockaddr_in6*>(&ss)->sin6_port);
    }
  }
  epfd_ = epoll_create1(EPOLL_CLOEXEC);
  evfd_ = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (epfd_ < 0 || evfd_ < 0) throw std::runtime_error("epoll/eventfd failed");
  epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.fd = listen_fd_;
  epoll_ctl(epfd_, EPOLL_CTL_ADD, listen_fd_, &ev);
  ev.events = EPOLLIN;
  ev.data.fd = evfd_;
  epoll_ctl(epfd_, EPOLL_CTL_ADD, evfd_, &ev);
  int nw = opts_.workers;
  if (nw <= 0) {
    int hw = int(std::thread::hardware_concurrency());
    nw = std::min(16, std::max(4, hw / 2));
  }
  pool_ = std::make_unique<WorkerPool>(nw);
  running_ = true;
  io_thread_ = std::thread([this] { io_loop(); });
}

void Server::wake() {
  uint64_t one = 1;
  ssize_t r = ::write(evfd_, &one, sizeof one);
  (void)r;
}

void Server::post_io(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lk(post_mu_);
    posted_.push_back(std::move(fn));
  }
  wake();
}

void Server::schedule_flush(const std::shared_ptr<Conn>& c) {
  {
    std::lock_guard<std::mutex> lk(post_mu_);
    flush_q_.push_back(c);
  }
  wake();
}

size_t Server::connections() const {
  std::lock_guard<std::mutex> lk(conns_mu_);
  return conns_.size();
}

void Server::stop(double grace_s) {
  if (!running_.load()) {
    if (pool_) pool_->stop();
    return;
  }
  stopping_ = true;
  // Stop accepting and close idle connections on the IO thread.
  post_io([this] {
    if (listen_fd_ >= 0) {
      epoll_ctl(epfd_, EPOLL_CTL_DEL, listen_fd_, nullptr);
      ::close(listen_fd_);
      listen_fd_ = -1;
    }
    std::vector<std::shared_ptr<Conn>> idle;
    {
      std::lock_guard<std::mutex> lk(conns_mu_);
      for (auto& [fd, c] : conns_)
        if (c->st != Conn::St::Dispatched) idle.push_back(c);
    }
    for (auto& c : idle) close_conn(c, false);
  });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(grace_s);
  while (inflight_.load() > 0 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  running_ = false;
  wake();
  if (io_thread_.joinable()) io_thread_.join();
  std::vector<std::shared_ptr<Conn>> all;
  {
    std::lock_guard<std::mutex> lk(conns_mu_);
    for (auto& [fd, c] : conns_) all.push_back(c);
  }
  for (auto& c : all) close_conn(c, true);
  if (pool_) pool_->stop();
  if (epfd_ >= 0) ::close(epfd_);
  if (evfd_ >= 0) ::close(evfd_);
  epfd_ = evfd_ = -1;
}

void Server::update_interest(const std::shared_ptr<Conn>& c) {
  if (c->st == Conn::St::Closed) return;
  uint32_t want = EPOLLRDHUP;
  bool kick = false;
  if (c->st == Conn::St::Handshake) {
    want |= EPOLLIN;
    std::lock_guard<std::mutex> lk(c->mu);
    if (c->tls_out_off < c->tls_out.size()) want |= EPOLLOUT;  // handshake records the socket did not take
  } else {
    // Read while parsing; while a request is in flight keep reading (to see EOF) unless a
    // large amount of pipelined input is buffered.
    if (c->st != Conn::St::Dispatched || c->in.size() < (1u << 20)) want |= EPOLLIN;
    std::lock_guard<std::mutex> lk(c->mu);
    // TLS: an SSL_write that wants to read waits for EPOLLIN (not EPOLLOUT, which would spin).
    if (c->want_out && c->pending_locked() > 0 && !c->tls_wr_want_read) want |= EPOLLOUT;
    if (c->tls_wr_want_read) want |= EPOLLIN;
#ifdef FE_HTTP_TLS
    // Decrypted bytes already inside SSL produce no epoll event: read them from the loop.
    if (c->ssl && (want & EPOLLIN) && SSL_pending(c->ssl) > 0) kick = true;
#endif
  }
  if (want != c->interest) {
    epoll_event ev{};
    ev.events = want;
    ev.data.fd = c->fd;
    epoll_ctl(epfd_, EPOLL_CTL_MOD, c->fd, &ev);
    c->interest = want;
  }
  if (kick) tls_kick_.push_back(c);
}

void Server::do_accept() {
  for (int i = 0; i < 256; ++i) {
    sockaddr_storage ss{};
    socklen_t sl = sizeof ss;
    int fd = ::accept4(listen_fd_, reinterpret_cast<sockaddr*>(&ss), &sl, SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EINTR) continue;
      return;  // EAGAIN or error (EMFILE...): try again on the next event
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    auto c = std::make_shared<Conn>();
    c->srv = this;
    c->fd = fd;
    c->id = next_conn_id_++;
    char host[INET6_ADDRSTRLEN] = "";
    int port = 0;
    if (ss.ss_family == AF_INET) {
      auto* a = reinterpret_cast<sockaddr_in*>(&ss);
      inet_ntop(AF_INET, &a->sin_addr, host, sizeof host);
      port = ntohs(a->sin_port);
    } else if (ss.ss_family == AF_INET6) {
      auto* a = reinterpret_cast<sockaddr_in6*>(&ss);
      inet_ntop(AF_INET6, &a->sin6_addr, host, sizeof host);
      port = ntohs(a->sin6_port);
    }
    c->client = std::string(host) + ":" + std::to_string(port);
    c->idle_since = std::chrono::steady_clock::now();
#ifdef FE_HTTP_TLS
    if (tls_ctx_) {
      ERR_clear_error();
      SSL* s = SSL_new(tls_ctx_);
      BIO* rb = s ? BIO_new_socket(fd, BIO_NOCLOSE) : nullptr;
      BIO* wb = rb ? BIO_new(BIO_s_mem()) : nullptr;
      if (!wb) {
        if (rb) BIO_free(rb);
        if (s) SSL_free(s);
        ERR_clear_error();
        ::close(fd);
        continue;
      }
      SSL_set_bio(s, rb, wb);  // SSL owns both
      SSL_set_accept_state(s);
      c->ssl = s;
      c->st = Conn::St::Handshake;
      c->hs_deadline = c->idle_since + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                           std::chrono::duration<double>(opts_.tls_handshake_timeout_s));
    }
#endif
    epoll_event ev{};
    ev.events = EPOLLIN | EPOLLRDHUP;
    ev.data.fd = fd;
    c->interest = ev.events;
    {
      std::lock_guard<std::mutex> lk(conns_mu_);
      conns_[fd] = c;
    }
    epoll_ctl(epfd_, EPOLL_CTL_ADD, fd, &ev);
  }
}

void Server::close_conn(const std::shared_ptr<Conn>& c, bool notify) {
  if (c->st == Conn::St::Closed) return;
  bool was_dispatched = c->st == Conn::St::Dispatched;
  c->st = Conn::St::Closed;
  if (epfd_ >= 0) epoll_ctl(epfd_, EPOLL_CTL_DEL, c->fd, nullptr);
  int fd;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    c->closed = true;
    fd = c->fd;
#ifdef FE_HTTP_TLS
    if (c->ssl) {
      // Best-effort close_notify after what is already encrypted (non-blocking, never waits for the
      // peer's); not after a fatal TLS or socket error.
      ERR_clear_error();
      if (c->tls_hs_done && !c->tls_fatal && !c->write_error) {
        SSL_shutdown(c->ssl);
        c->tls_collect_locked();
      }
      if (!c->write_error) c->tls_send_locked();
      ERR_clear_error();
      SSL_free(c->ssl);
      c->ssl = nullptr;
    }
#endif
    ::close(c->fd);
  }
  {
    std::lock_guard<std::mutex> lk(conns_mu_);
    auto it = conns_.find(fd);
    if (it != conns_.end() && it->second == c) conns_.erase(it);
  }
  if (c->resp) {
    auto r = std::move(c->resp);
    if (was_dispatched) inflight_--;
    if (!r->finished() || notify) r->fire_disconnect();
    else r->disconnected_.store(true);
  }
}

void Server::send_simple_and_close(const std::shared_ptr<Conn>& c, int status, std::string_view ctype,
                                   std::string_view body) {
  std::string data = "HTTP/1.1 " + std::to_string(status) + " " + std::string(status_phrase(status)) + "\r\n";
  data += default_headers();
  data += "content-type: ";
  data += ctype;
  data += "\r\ncontent-length: " + std::to_string(body.size()) + "\r\nconnection: close\r\n\r\n";
  data += body;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    c->out += data;
    c->flush_locked();
  }
  close_conn(c, true);
}

#ifdef FE_HTTP_TLS
void Server::tls_handshake(const std::shared_ptr<Conn>& c) {
  int r = 0, e = SSL_ERROR_SYSCALL;
  bool sock_err = false;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    if (!c->tls_send_locked()) {  // earlier handshake records first
      sock_err = true;
    } else {
      ERR_clear_error();
      r = SSL_do_handshake(c->ssl);
      e = r == 1 ? SSL_ERROR_NONE : SSL_get_error(c->ssl, r);
      if (r == 1) c->tls_hs_done = true;
      if (e == SSL_ERROR_SSL || e == SSL_ERROR_SYSCALL) c->tls_fatal = true;
      ERR_clear_error();
      c->tls_collect_locked();  // ServerHello..Finished, tickets, or an alert
      if (!c->tls_send_locked()) sock_err = true;
      if (r == 1 && c->tls_out_off < c->tls_out.size()) c->want_out = true;  // tickets: EPOLLOUT
    }
  }
  if (r == 1 && !sock_err) {
    tls_ok_++;
    c->st = Conn::St::Headers;
    c->idle_since = std::chrono::steady_clock::now();
    on_readable(c);  // the first request may have arrived with the client's Finished
    return;
  }
  if (!sock_err && (e == SSL_ERROR_WANT_READ || e == SSL_ERROR_WANT_WRITE)) {
    update_interest(c);
    return;
  }
  // Not TLS (plain HTTP, garbage), a version / cipher mismatch, or EOF: just close, no response.
  tls_fail_++;
  close_conn(c, false);
}
#else
void Server::tls_handshake(const std::shared_ptr<Conn>& c) { close_conn(c, false); }
#endif

void Server::on_readable(const std::shared_ptr<Conn>& c) {
  char buf[65536];
  size_t total = 0;
  for (;;) {
#ifdef FE_HTTP_TLS
    if (c->ssl) {
      int r, e;
      {
        std::lock_guard<std::mutex> lk(c->mu);
        ERR_clear_error();
        r = SSL_read(c->ssl, buf, int(sizeof buf));
        e = r > 0 ? SSL_ERROR_NONE : SSL_get_error(c->ssl, r);
        if (e == SSL_ERROR_SSL || e == SSL_ERROR_SYSCALL) c->tls_fatal = true;
        if (r <= 0) ERR_clear_error();
        c->tls_after_io_locked();  // e.g. our KeyUpdate reply, a no_renegotiation alert
      }
      if (r > 0) {
        c->in.append(buf, size_t(r));
        total += size_t(r);
        if (total >= (4u << 20)) break;  // fairness; leftovers inside SSL are kicked (update_interest)
        continue;
      }
      if (e == SSL_ERROR_WANT_READ) break;
      // close_notify, EOF without close_notify, reset or a TLS error: like the plain EOF path.
      try_parse(c);
      close_conn(c, true);
      return;
    }
#endif
    ssize_t n = ::recv(c->fd, buf, sizeof buf, 0);
    if (n > 0) {
      c->in.append(buf, size_t(n));
      total += size_t(n);
      if (total >= (4u << 20)) break;  // fairness; level-triggered epoll calls us again
      continue;
    }
    if (n == 0) {
      // EOF: uvicorn closes the transport (eof_received returns None) -> disconnect.
      try_parse(c);
      close_conn(c, true);
      return;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    close_conn(c, true);
    return;
  }
  try_parse(c);
  if (c->st == Conn::St::Closed) return;
  if (c->ssl) {
    bool retry_write;
    {
      std::lock_guard<std::mutex> lk(c->mu);
      retry_write = c->tls_wr_want_read && c->pending_locked() > 0;
      c->tls_wr_want_read = false;
    }
    if (retry_write) {  // an SSL_write waited for this input
      on_writable_ev(c);
      return;
    }
  }
  update_interest(c);
}

void Server::try_parse(const std::shared_ptr<Conn>& c) {
  for (;;) {
    if (c->st == Conn::St::Closed || c->st == Conn::St::Dispatched) return;
    if (c->st == Conn::St::Headers) {
      // Skip leading CRLFs between requests (RFC 9112 2.2).
      size_t lead = 0;
      while (lead < c->in.size() && (c->in[lead] == '\r' || c->in[lead] == '\n')) ++lead;
      if (lead) c->in.erase(0, lead);
      if (c->in.empty()) return;
      size_t end = c->in.find("\r\n\r\n");
      size_t sep = 4;
      if (end == std::string::npos) {
        size_t e2 = c->in.find("\n\n");
        if (e2 != std::string::npos) {
          end = e2;
          sep = 2;
        }
      }
      if (end == std::string::npos) {
        if (c->in.size() > opts_.max_header_bytes) {
          send_simple_and_close(c, 400, "text/plain; charset=utf-8", "Invalid HTTP request received.");
        }
        return;
      }
      std::string_view head(c->in.data(), end);
      auto req = std::make_shared<Request>();
      req->client = c->client;
      req->conn_id = c->id;
      req->received = std::chrono::steady_clock::now();
      // request line
      size_t le = head.find('\n');
      std::string_view line = head.substr(0, le);
      if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
      size_t sp1 = line.find(' ');
      size_t sp2 = sp1 == std::string_view::npos ? sp1 : line.find(' ', sp1 + 1);
      bool bad = sp1 == std::string_view::npos || sp2 == std::string_view::npos || sp1 == 0;
      std::string_view ver;
      if (!bad) {
        req->method = std::string(line.substr(0, sp1));
        req->target = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
        ver = line.substr(sp2 + 1);
        if (ver == "HTTP/1.1")
          req->http_version = "1.1";
        else if (ver == "HTTP/1.0")
          req->http_version = "1.0";
        else
          bad = true;
        for (char ch : req->method)
          if (!(ch >= 'A' && ch <= 'Z') && ch != '-' && ch != '_') bad = true;
        if (req->target.empty()) bad = true;
      }
      // headers
      bool chunked = false, have_cl = false, ka_close = false, ka_keep = false, expect100 = false;
      size_t cl = 0;
      std::string_view rest = le == std::string_view::npos ? std::string_view() : head.substr(le + 1);
      while (!bad && !rest.empty()) {
        size_t e = rest.find('\n');
        std::string_view hl = rest.substr(0, e);
        rest = e == std::string_view::npos ? std::string_view() : rest.substr(e + 1);
        if (!hl.empty() && hl.back() == '\r') hl.remove_suffix(1);
        if (hl.empty()) continue;
        size_t colon = hl.find(':');
        if (colon == std::string_view::npos || colon == 0) {
          bad = true;
          break;
        }
        std::string name = to_lower(hl.substr(0, colon));
        for (char ch : name)
          if (ch == ' ' || ch == '\t') bad = true;
        std::string value(trim_ows(hl.substr(colon + 1)));
        if (name == "content-length") {
          if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos ||
              value.size() > 18) {
            bad = true;
            break;
          }
          size_t v = std::stoull(value);
          if (have_cl && v != cl) bad = true;
          cl = v;
          have_cl = true;
        } else if (name == "transfer-encoding") {
          std::string lv = to_lower(value);
          if (lv.size() >= 7 && lv.compare(lv.size() - 7, 7, "chunked") == 0) chunked = true;
          else bad = true;  // unsupported transfer coding
        } else if (name == "connection") {
          std::string lv = to_lower(value);
          if (lv.find("close") != std::string::npos) ka_close = true;
          if (lv.find("keep-alive") != std::string::npos) ka_keep = true;
        } else if (name == "expect") {
          if (ieq(value, "100-continue")) expect100 = true;
        }
        req->headers.emplace_back(std::move(name), std::move(value));
        if (req->headers.size() > opts_.max_header_count) bad = true;
      }
      if (chunked && have_cl) bad = true;
      if (bad) {
        send_simple_and_close(c, 400, "text/plain; charset=utf-8", "Invalid HTTP request received.");
        return;
      }
      // keep-alive semantics: llhttp should_keep_alive + uvicorn's "1.0 -> close"
      req->keep_alive = req->http_version == "1.1" ? !ka_close : false;
      (void)ka_keep;
      // path / query
      std::string_view tgt = req->target;
      size_t q = tgt.find('?');
      std::string_view rp = tgt.substr(0, q);
      if (q != std::string_view::npos) req->query = std::string(tgt.substr(q + 1));
      size_t hash = req->query.find('#');
      if (hash != std::string::npos) req->query.resize(hash);
      // absolute-form "http://host/path"
      if (rp.size() > 7 && (rp.substr(0, 7) == "http://" || rp.substr(0, 8) == "https://")) {
        size_t s = rp.find('/', rp.find("//") + 2);
        rp = s == std::string_view::npos ? std::string_view("/") : rp.substr(s);
      }
      size_t h2 = rp.find('#');
      if (h2 != std::string_view::npos) rp = rp.substr(0, h2);
      req->raw_path = std::string(rp);
      req->path = req->raw_path.find('%') != std::string::npos ? percent_decode(req->raw_path, false)
                                                                : req->raw_path;
      c->in.erase(0, end + sep);
      c->req = req;
      c->had_request = true;
      if (have_cl && cl > opts_.max_body_bytes) {
        send_simple_and_close(c, 413, "text/plain; charset=utf-8", "Request Entity Too Large");
        return;
      }
      if (expect100 && (chunked || cl > 0)) {
        std::lock_guard<std::mutex> lk(c->mu);
        c->out += "HTTP/1.1 100 Continue\r\n\r\n";
        c->flush_locked();
      }
      if (chunked) {
        c->st = Conn::St::Chunked;
        c->ch = Conn::Ch::Size;
      } else if (cl > 0) {
        c->st = Conn::St::Body;
        c->body_left = cl;
        req->body.reserve(cl);
      } else {
        c->st = Conn::St::Dispatched;
      }
    }
    if (c->st == Conn::St::Body) {
      size_t take = std::min(c->body_left, c->in.size());
      c->req->body.append(c->in, 0, take);
      c->in.erase(0, take);
      c->body_left -= take;
      if (c->body_left > 0) return;
      c->st = Conn::St::Dispatched;
    }
    if (c->st == Conn::St::Chunked) {
      bool done = false;
      while (!done) {
        if (c->ch == Conn::Ch::Size) {
          size_t e = c->in.find('\n');
          if (e == std::string::npos) {
            if (c->in.size() > 4096) {
              send_simple_and_close(c, 400, "text/plain; charset=utf-8", "Invalid HTTP request received.");
              return;
            }
            return;
          }
          std::string_view sl(c->in.data(), e);
          if (!sl.empty() && sl.back() == '\r') sl.remove_suffix(1);
          size_t semi = sl.find(';');
          sl = trim_ows(sl.substr(0, semi));
          size_t v = 0;
          bool okhex = !sl.empty() && sl.size() <= 15;
          for (char ch : sl) {
            int h = hexval(ch);
            if (h < 0) {
              okhex = false;
              break;
            }
            v = v * 16 + size_t(h);
          }
          if (!okhex) {
            send_simple_and_close(c, 400, "text/plain; charset=utf-8", "Invalid HTTP request received.");
            return;
          }
          c->in.erase(0, e + 1);
          if (v == 0) {
            c->ch = Conn::Ch::Trailer;
          } else {
            if (c->req->body.size() + v > opts_.max_body_bytes) {
              send_simple_and_close(c, 413, "text/plain; charset=utf-8", "Request Entity Too Large");
              return;
            }
            c->chunk_left = v;
            c->ch = Conn::Ch::Data;
          }
        } else if (c->ch == Conn::Ch::Data) {
          size_t take = std::min(c->chunk_left, c->in.size());
          c->req->body.append(c->in, 0, take);
          c->in.erase(0, take);
          c->chunk_left -= take;
          if (c->chunk_left > 0) return;
          c->ch = Conn::Ch::DataEnd;
        } else if (c->ch == Conn::Ch::DataEnd) {
          if (c->in.size() < 1) return;
          if (c->in[0] == '\n') {
            c->in.erase(0, 1);
          } else {
            if (c->in.size() < 2) return;
            if (c->in[0] != '\r' || c->in[1] != '\n') {
              send_simple_and_close(c, 400, "text/plain; charset=utf-8", "Invalid HTTP request received.");
              return;
            }
            c->in.erase(0, 2);
          }
          c->ch = Conn::Ch::Size;
        } else {  // Trailer: lines until an empty line
          size_t e = c->in.find('\n');
          if (e == std::string::npos) return;
          std::string_view tl(c->in.data(), e);
          if (!tl.empty() && tl.back() == '\r') tl.remove_suffix(1);
          c->in.erase(0, e + 1);
          if (tl.empty()) done = true;
        }
      }
      c->st = Conn::St::Dispatched;
    }
    if (c->st == Conn::St::Dispatched) {
      auto req = c->req;
      c->req.reset();
      auto resp = std::shared_ptr<Response>(new Response(c, req->method == "HEAD", req->keep_alive));
      if (opts_.access_log)
        resp->access_line_ = req->client + " - \"" + req->method + " " + req->target + " HTTP/" + req->http_version + "\"";
      c->resp = resp;
      {
        std::lock_guard<std::mutex> lk(c->mu);
        c->resp_done = false;
        c->congested = false;
      }
      inflight_++;
      pool_->post([this, req, resp] {
        try {
          handler_(req, resp);
        } catch (const std::exception& e) {
          fprintf(stderr, "[fe_http] handler threw: %s\n", e.what());
          if (!resp->started())
            resp->send(500, {{"content-type", "text/plain; charset=utf-8"}, {"connection", "close"}},
                       "Internal Server Error");
        }
      });
      return;
    }
  }
}

void Server::response_complete(const std::shared_ptr<Conn>& c) {
  if (c->st != Conn::St::Dispatched) return;
  bool ka = c->resp ? c->resp->keep_alive_ : false;
  c->resp.reset();
  inflight_--;
  if (!ka || stopping_.load()) {
    close_conn(c, false);
    return;
  }
  c->st = Conn::St::Headers;
  c->idle_since = std::chrono::steady_clock::now();
  try_parse(c);
  if (c->st != Conn::St::Closed) update_interest(c);
}

void Server::on_writable_ev(const std::shared_ptr<Conn>& c) {
  bool complete = false, err = false, fire_w = false;
  {
    std::lock_guard<std::mutex> lk(c->mu);
    if (c->closed) return;
    if (!c->flush_locked()) err = true;
    if (c->pending_locked() == 0) {
      c->want_out = false;
      if (c->resp_done) {
        complete = true;
        c->resp_done = false;
      }
    }
    if (c->congested && c->pending_locked() <= opts_.low_water) {
      c->congested = false;
      fire_w = true;
    }
  }
  if (err) {
    close_conn(c, true);
    return;
  }
  if (fire_w && c->resp) c->resp->fire_writable();
  if (complete) {
    response_complete(c);
    return;
  }
  update_interest(c);
}

void Server::io_loop() {
  epoll_event evs[256];
  auto last_sweep = std::chrono::steady_clock::now();
  while (running_.load()) {
    int n = epoll_wait(epfd_, evs, 256, tls_kick_.empty() ? 250 : 0);
    for (int i = 0; i < n; ++i) {
      int fd = evs[i].data.fd;
      uint32_t e = evs[i].events;
      if (fd == listen_fd_) {
        if (!stopping_) do_accept();
        continue;
      }
      if (fd == evfd_) {
        uint64_t v;
        while (::read(evfd_, &v, sizeof v) > 0) {
        }
        continue;
      }
      std::shared_ptr<Conn> c;
      {
        std::lock_guard<std::mutex> lk(conns_mu_);
        auto it = conns_.find(fd);
        if (it != conns_.end()) c = it->second;
      }
      if (!c) continue;
      if (e & (EPOLLERR | EPOLLHUP)) {
        if (c->st == Conn::St::Handshake) tls_fail_++;
        close_conn(c, true);
        continue;
      }
      if (c->st == Conn::St::Handshake) {  // TLS: every event drives the handshake
        tls_handshake(c);
        continue;
      }
      if (e & EPOLLOUT) on_writable_ev(c);
      if (c->st == Conn::St::Closed) continue;
      if (e & EPOLLIN) {
        on_readable(c);
      } else if (e & EPOLLRDHUP) {
        // Peer shut down its write side while we are not reading: disconnect.
        close_conn(c, true);
      }
    }
    // posted closures and flushes
    std::vector<std::function<void()>> posted;
    std::vector<std::weak_ptr<Conn>> flushes;
    {
      std::lock_guard<std::mutex> lk(post_mu_);
      posted.swap(posted_);
      flushes.swap(flush_q_);
    }
    for (auto& w : flushes)
      if (auto c = w.lock())
        if (c->st != Conn::St::Closed) on_writable_ev(c);
    for (auto& f : posted) f();
    // TLS: connections with decrypted input still inside SSL (no epoll event will come for it)
    if (!tls_kick_.empty()) {
      std::vector<std::weak_ptr<Conn>> kicks;
      kicks.swap(tls_kick_);
      for (auto& w : kicks)
        if (auto c = w.lock())
          if (c->st != Conn::St::Closed && c->st != Conn::St::Handshake && (c->interest & EPOLLIN)) on_readable(c);
    }
    // keep-alive sweep
    auto now = std::chrono::steady_clock::now();
    if (now - last_sweep > std::chrono::milliseconds(200)) {
      last_sweep = now;
      std::vector<std::shared_ptr<Conn>> expired, hs_expired;
      {
        std::lock_guard<std::mutex> lk(conns_mu_);
        for (auto& [cfd, c] : conns_) {
          if (c->st == Conn::St::Handshake) {  // TLS handshake allowance (slow / stuck clients)
            if (now > c->hs_deadline) hs_expired.push_back(c);
            continue;
          }
          if (c->st != Conn::St::Headers || !c->in.empty()) continue;
          double lim = c->had_request ? opts_.keep_alive_timeout_s : opts_.first_request_timeout_s;
          if (std::chrono::duration<double>(now - c->idle_since).count() > lim) expired.push_back(c);
        }
      }
      for (auto& c : expired) close_conn(c, false);
      for (auto& c : hs_expired) {
        tls_timeout_++;
        close_conn(c, false);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------

std::string_view status_phrase(int s) {
  switch (s) {
    case 100: return "Continue";
    case 101: return "Switching Protocols";
    case 200: return "OK";
    case 201: return "Created";
    case 202: return "Accepted";
    case 204: return "No Content";
    case 206: return "Partial Content";
    case 301: return "Moved Permanently";
    case 302: return "Found";
    case 304: return "Not Modified";
    case 307: return "Temporary Redirect";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 403: return "Forbidden";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 406: return "Not Acceptable";
    case 408: return "Request Timeout";
    case 409: return "Conflict";
    case 411: return "Length Required";
    case 413: return "Request Entity Too Large";
    case 415: return "Unsupported Media Type";
    case 422: return "Unprocessable Entity";
    case 429: return "Too Many Requests";
    case 499: return "";
    case 500: return "Internal Server Error";
    case 501: return "Not Implemented";
    case 502: return "Bad Gateway";
    case 503: return "Service Unavailable";
    case 504: return "Gateway Timeout";
    default: return "";
  }
}

bool constant_time_eq(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  unsigned char d = 0;
  for (size_t i = 0; i < a.size(); ++i) d |= static_cast<unsigned char>(a[i] ^ b[i]);
  return d == 0;
}

// SHA-256 (FIPS 180-4), straightforward implementation.
std::string sha256(std::string_view data) {
  static const uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  std::string m(data);
  uint64_t bitlen = uint64_t(data.size()) * 8;
  m.push_back(char(0x80));
  while (m.size() % 64 != 56) m.push_back(0);
  for (int i = 7; i >= 0; --i) m.push_back(char((bitlen >> (i * 8)) & 0xff));
  for (size_t off = 0; off < m.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t(uint8_t(m[off + 4 * i])) << 24) | (uint32_t(uint8_t(m[off + 4 * i + 1])) << 16) |
             (uint32_t(uint8_t(m[off + 4 * i + 2])) << 8) | uint32_t(uint8_t(m[off + 4 * i + 3]));
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = S0 + mj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
  }
  std::string out(32, '\0');
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j) out[size_t(i * 4 + j)] = char((h[i] >> (24 - 8 * j)) & 0xff);
  return out;
}

}  // namespace fe::http
