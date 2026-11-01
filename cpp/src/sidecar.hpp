// Connection to the Python sidecar (vrsim/sidecar.py): JSON lines over localhost TCP.
// A background thread (re)connects and queues incoming messages; the frame loop drains them
// between frames and sends small events (push-to-talk, spawn results).
#pragma once

#include <nlohmann/json.hpp>

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

namespace vrsim
{

class SidecarLink
{
public:
    explicit SidecarLink(int port);
    ~SidecarLink();

    bool connected() const { return m_connected; }
    bool poll(nlohmann::json& out);  // next queued message, if any
    void send(const nlohmann::json& message);

private:
    void run();

    int m_port;
    std::atomic<bool> m_running{ true };
    std::atomic<bool> m_connected{ false };
    std::atomic<unsigned long long> m_socket;
    std::mutex m_lock;
    std::deque<nlohmann::json> m_inbox;
    std::thread m_thread;
};

} // namespace vrsim
