// fe_http: a small HTTP/1.1 server (epoll IO thread + worker pool) that mimics the
// client-visible behaviour of uvicorn 0.54 (httptools) as used by vLLM 0.32.0.
//
// - Adopts an inherited listening fd (--listen-fd) or binds 127.0.0.1:<port> (tests).
// - Keep-alive with uvicorn's idle timeout (VLLM_HTTP_TIMEOUT_KEEP_ALIVE, default 5 s).
// - Content-Length and chunked request bodies; "Expect: 100-continue".
// - Responses: "date" + "server: uvicorn" default headers first, then the app's headers,
//   lower-case names, "transfer-encoding: chunked" added when no content-length is given.
// - Client disconnect (EOF, RST, RDHUP) fires the response's disconnect callback once.
// - Backpressure: Response::pending_bytes() and an on_writable callback (low-water mark).
//
// Thread model: one IO thread owns all sockets' read side and parsing. Any thread may call
// Response methods; they append to the connection's output buffer under its mutex and try
// a direct non-blocking write; leftovers are flushed by the IO thread on EPOLLOUT.
//
// TLS (HTTPS, builds with FE_HTTP_TLS = OpenSSL found): ServerOptions::tls_certfile turns it on for
// every connection of the server. Per-connection SSL on the same non-blocking fd: SSL reads the socket
// (WANT_READ -> wait for EPOLLIN) and writes its records into a memory BIO whose bytes go through the
// connection's out path (send MSG_NOSIGNAL, EPOLLOUT, pending_bytes / congested / on_writable unchanged;
// pending counts plaintext not yet encrypted + ciphertext not yet sent). The handshake runs in the IO
// thread (state Handshake, closed after tls_handshake_timeout_s); afterwards SSL_read (IO thread) and
// SSL_write (any thread) are serialised by the connection's mutex. TLS >= 1.2, ALPN http/1.1, no
// renegotiation, TLS 1.3 KeyUpdate answered in-stream, close_notify on close. A failed handshake (plain
// HTTP, garbage, timeout) just closes the connection.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

struct ssl_ctx_st;  // OpenSSL's SSL_CTX (only used when built with FE_HTTP_TLS)

namespace fe::http {

using HeaderList = std::vector<std::pair<std::string, std::string>>;

struct Request {
  std::string method;
  std::string target;    // raw request-target
  std::string raw_path;  // path part of the target, undecoded
  std::string path;      // percent-decoded like uvicorn (only when '%' present)
  std::string query;     // raw query string (without '?')
  std::string http_version = "1.1";
  bool keep_alive = true;
  HeaderList headers;  // names lower-cased (ASGI scope semantics), values verbatim
  std::string body;
  std::string client;  // "ip:port"
  uint64_t conn_id = 0;
  std::chrono::steady_clock::time_point received;
  // First header with this (lower-case) name, or nullptr.
  const std::string* header(std::string_view lname) const;
  // Query parameter value (first occurrence, percent/plus decoded), or nullptr.
  std::optional<std::string> query_param(std::string_view name) const;
};

class Conn;
class Server;

// Per-request response handle. Thread-safe. Exactly one of send() or start()+end().
class Response : public std::enable_shared_from_this<Response> {
 public:
  ~Response();
  // Complete response with a body; adds content-length.
  void send(int status, HeaderList headers, std::string body);
  // Starts a streaming (chunked) response.
  void start(int status, HeaderList headers);
  // Writes one body piece of a streaming response. Returns false if the client is gone.
  bool write(std::string_view data);
  // Ends a streaming response.
  void end();
  bool disconnected() const { return disconnected_.load(std::memory_order_acquire); }
  bool started() const { return started_.load(std::memory_order_acquire); }
  bool finished() const { return finished_.load(std::memory_order_acquire); }
  // Bytes queued in user space for this connection and not yet written to the socket.
  size_t pending_bytes() const;
  // Called once (on a worker thread) when the client disconnects before end()/send()
  // completed. If the client was already gone when set, it is called immediately.
  void on_disconnect(std::function<void()> cb);
  // Called (on a worker thread) whenever pending_bytes() drops below the low-water mark
  // after having been above the high-water mark; also once when the client disconnects (the
  // callback then sees disconnected()), so a writer parked on congested() always wakes up.
  void on_writable(std::function<void()> cb);
  bool congested() const;

 private:
  friend class Conn;
  friend class Server;
  explicit Response(std::shared_ptr<Conn> c, bool head, bool keep_alive);
  void fire_disconnect();
  void fire_writable();
  std::string build_head(int status, const HeaderList& headers, bool has_length, size_t length,
                         bool* chunked_out);
  std::weak_ptr<Conn> conn_;
  std::string access_line_;  // "<client> - \"<METHOD> <target> HTTP/<v>\"" when the access log is on
  bool head_method_ = false;
  bool keep_alive_ = true;
  bool chunked_ = false;
  std::atomic<bool> started_{false}, finished_{false}, disconnected_{false};
  std::mutex cb_mu_;
  std::function<void()> on_disconnect_, on_writable_;
  bool disconnect_fired_ = false;
};

using Handler = std::function<void(std::shared_ptr<Request>, std::shared_ptr<Response>)>;

struct ServerOptions {
  int listen_fd = -1;                 // adopt this listening socket if >= 0
  std::string host = "127.0.0.1";     // else bind host:port
  int port = 0;
  int workers = 0;                    // 0 -> min(16, max(4, hw/2))
  double keep_alive_timeout_s = 5.0;  // uvicorn timeout_keep_alive
  double first_request_timeout_s = 300.0;  // idle connections that never sent a request
  size_t max_header_bytes = 4 << 20;  // h11_max_incomplete_event_size default
  size_t max_header_count = 256;
  size_t max_body_bytes = size_t(1) << 30;  // 1 GiB; uvicorn has no limit
  size_t high_water = 1 << 20;        // per-connection output backpressure
  size_t low_water = 256 << 10;
  bool date_header = true;
  bool server_header = true;
  // uvicorn's access log on stderr: INFO:     <client> - "<METHOD> <target> HTTP/<v>" <status> <phrase>
  bool access_log = false;
  // ---- TLS (vLLM's --ssl-* args). Enabled when tls_certfile is non-empty; start() throws if the
  // build has no TLS (Server::tls_available()) or the files cannot be loaded. Error messages name the
  // cert / CA paths but never the key path, and never file contents.
  std::string tls_certfile;   // --ssl-certfile: PEM certificate chain (leaf first)
  std::string tls_keyfile;    // --ssl-keyfile: PEM private key (unencrypted); empty -> the key is in tls_certfile
  std::string tls_ca_certs;   // --ssl-ca-certs: CA bundle for client-certificate verification
  int tls_cert_reqs = 0;      // --ssl-cert-reqs: 0 CERT_NONE, 1 CERT_OPTIONAL, 2 CERT_REQUIRED
  std::string tls_ciphers;    // --ssl-ciphers (TLS <= 1.2); empty -> Python's ssl default list
  double tls_handshake_timeout_s = 10.0;  // un-finished handshakes are closed after this
};

// TLS counters (monotonic; for tests and logs).
struct TlsStats {
  uint64_t handshakes_ok = 0;
  uint64_t handshake_failures = 0;  // protocol errors: plain HTTP, garbage, version / cipher mismatch, EOF
  uint64_t handshake_timeouts = 0;
};

// Simple fixed-size worker pool (FIFO).
class WorkerPool {
 public:
  explicit WorkerPool(int n);
  ~WorkerPool();
  void post(std::function<void()> fn);
  void stop();
  size_t queued() const;

 private:
  void run();
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> q_;
  std::vector<std::thread> threads_;
  bool stop_ = false;
};

class Server {
 public:
  Server(ServerOptions opts, Handler handler);
  ~Server();
  // Binds / adopts the socket and starts the IO thread and workers. Throws on failure.
  void start();
  // Bound port (useful with port 0).
  int port() const { return bound_port_; }
  // Graceful stop: stop accepting, close idle connections, wait up to grace_s for in-flight
  // responses, then close everything and join threads.
  void stop(double grace_s = 2.0);
  // Number of open connections / in-flight requests.
  size_t connections() const;
  size_t inflight() const { return inflight_.load(); }
  WorkerPool& pool() { return *pool_; }
  // Thread-safe: run fn on the IO thread.
  void post_io(std::function<void()> fn);
  // True when this build has TLS (OpenSSL) compiled in.
  static bool tls_available();
  // Loads o's certificate / key / CA files into a throwaway context: throws std::runtime_error with
  // the same messages start() would (fail fast before the engine is up). Never names the key path.
  static void check_tls(const ServerOptions& o);
  bool tls_enabled() const { return tls_ctx_ != nullptr; }
  TlsStats tls_stats() const;

 private:
  friend class Conn;
  friend class Response;
  void io_loop();
  void do_accept();
  void tls_init();                                       // builds tls_ctx_ from opts_ (throws)
  void tls_handshake(const std::shared_ptr<Conn>& c);    // IO thread, state Handshake
  void wake();
  void schedule_flush(const std::shared_ptr<Conn>& c);
  void close_conn(const std::shared_ptr<Conn>& c, bool notify);
  void on_readable(const std::shared_ptr<Conn>& c);
  void on_writable_ev(const std::shared_ptr<Conn>& c);
  void try_parse(const std::shared_ptr<Conn>& c);
  void response_complete(const std::shared_ptr<Conn>& c);
  void update_interest(const std::shared_ptr<Conn>& c);
  void send_simple_and_close(const std::shared_ptr<Conn>& c, int status, std::string_view ctype,
                             std::string_view body);
  std::string default_headers();

  ServerOptions opts_;
  Handler handler_;
  int listen_fd_ = -1;
  int epfd_ = -1;
  int evfd_ = -1;
  int bound_port_ = 0;
  std::atomic<bool> running_{false};
  std::atomic<bool> stopping_{false};
  std::thread io_thread_;
  std::unique_ptr<WorkerPool> pool_;
  mutable std::mutex conns_mu_;
  std::unordered_map<int, std::shared_ptr<Conn>> conns_;  // by fd (IO thread mutates)
  std::mutex post_mu_;
  std::vector<std::function<void()>> posted_;
  std::vector<std::weak_ptr<Conn>> flush_q_;
  std::atomic<size_t> inflight_{0};
  std::atomic<uint64_t> next_conn_id_{1};
  // TLS: SSL_CTX (opaque here so that this header does not need OpenSSL's)
  ssl_ctx_st* tls_ctx_ = nullptr;
  std::vector<std::weak_ptr<Conn>> tls_kick_;  // IO thread: connections with decrypted input left in SSL
  std::atomic<uint64_t> tls_ok_{0}, tls_fail_{0}, tls_timeout_{0};
  // cached date header
  std::mutex date_mu_;
  int64_t date_sec_ = -1;
  std::string date_str_;
};

// Reason phrase of Python 3.12's http.HTTPStatus (what uvicorn writes on the status line).
std::string_view status_phrase(int status);

// Constant-time compare of SHA-256 digests (the vLLM auth middleware's check).
std::string sha256(std::string_view data);
bool constant_time_eq(std::string_view a, std::string_view b);

}  // namespace fe::http
