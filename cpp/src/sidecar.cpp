#include "sidecar.hpp"

#include "common.hpp"

#include <winsock2.h>
#include <ws2tcpip.h>

#include <chrono>

namespace vrsim
{

SidecarLink::SidecarLink(int port) : m_port(port), m_socket(INVALID_SOCKET)
{
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
        die("WSAStartup");
    m_thread = std::thread([this] { run(); });
}

SidecarLink::~SidecarLink()
{
    m_running = false;
    const SOCKET s = static_cast<SOCKET>(m_socket.exchange(INVALID_SOCKET));
    if (s != INVALID_SOCKET)
        closesocket(s);  // unblocks recv
    if (m_thread.joinable())
        m_thread.join();
    WSACleanup();
}

bool SidecarLink::poll(nlohmann::json& out)
{
    std::lock_guard<std::mutex> guard(m_lock);
    if (m_inbox.empty())
        return false;
    out = std::move(m_inbox.front());
    m_inbox.pop_front();
    return true;
}

void SidecarLink::send(const nlohmann::json& message)
{
    const SOCKET s = static_cast<SOCKET>(m_socket.load());
    if (s == INVALID_SOCKET)
        return;
    const std::string line = message.dump() + "\n";
    ::send(s, line.data(), static_cast<int>(line.size()), 0);
}

void SidecarLink::run()
{
    bool warned = false;
    while (m_running)
    {
        SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<u_short>(m_port));
        inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
        if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
        {
            closesocket(s);
            if (!warned)
                logf("Sidecar not reachable on 127.0.0.1:%d; start it with: uv run python -m vrsim.sidecar", m_port);
            warned = true;
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        BOOL noDelay = TRUE;
        setsockopt(s, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&noDelay), sizeof(noDelay));
        m_socket = static_cast<unsigned long long>(s);
        m_connected = true;
        warned = false;
        logf("Connected to sidecar");

        std::string buffer;
        char chunk[65536];
        for (;;)
        {
            const int n = recv(s, chunk, sizeof(chunk), 0);
            if (n <= 0)
                break;
            buffer.append(chunk, n);
            size_t nl;
            while ((nl = buffer.find('\n')) != std::string::npos)
            {
                const std::string line = buffer.substr(0, nl);
                buffer.erase(0, nl + 1);
                if (line.empty())
                    continue;
                try
                {
                    std::lock_guard<std::mutex> guard(m_lock);
                    m_inbox.push_back(nlohmann::json::parse(line));
                }
                catch (const std::exception& e)
                {
                    logf("Bad sidecar message: %s", e.what());
                }
            }
        }
        m_connected = false;
        const SOCKET old = static_cast<SOCKET>(m_socket.exchange(INVALID_SOCKET));
        if (old != INVALID_SOCKET)
            closesocket(old);
        if (m_running)
            logf("Sidecar disconnected; reconnecting...");
    }
}

} // namespace vrsim
