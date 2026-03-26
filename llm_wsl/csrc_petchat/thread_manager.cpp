#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <condition_variable>
#include <queue>
#include <functional>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#endif

namespace py = pybind11;

static bool set_process_affinity_impl(const std::vector<int> &cpus)
{
#ifdef __linux__
    if (cpus.empty())
        return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
    {
        if (c >= 0)
            CPU_SET(c, &set);
    }
    int rc = sched_setaffinity(0, sizeof(set), &set);
    return rc == 0;
#else
    (void)cpus;
    return false;
#endif
}

static std::vector<int> get_process_affinity_impl()
{
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    std::vector<int> out;
    int rc = sched_getaffinity(0, sizeof(set), &set);
    if (rc != 0)
        return out;
    int n = CPU_COUNT(&set);
    if (n <= 0)
        return out;
    for (int i = 0; i < 1024; ++i)
    {
        if (CPU_ISSET(i, &set))
            out.push_back(i);
    }
    return out;
#else
    return {};
#endif
}

class ThreadManager
{
public:
    ThreadManager(int num_threads) : stop(false)
    {
        for (int i = 0; i < num_threads; ++i)
        {
            workers.emplace_back([this]
                                 {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                } });
        }
        std::cout << "[C++ ThreadManager] Initialized with " << num_threads << " worker threads." << std::endl;
    }

    ~ThreadManager()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
        std::cout << "[C++ ThreadManager] All threads joined and manager destroyed." << std::endl;
    }

    int num_workers() const
    {
        return static_cast<int>(workers.size());
    }

    void add_task(int task_id, int complexity)
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace([task_id, complexity]
                          {
                              long long sum = 0;
                              for (int i = 0; i < complexity; ++i)
                              {
                                  sum += i * i;
                              } });
        }
        condition.notify_one();
    }

    std::vector<torch::Tensor> parallel_process(std::vector<torch::Tensor> inputs)
    {
        std::vector<torch::Tensor> results(inputs.size());
        std::vector<std::thread> batch_threads;
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            batch_threads.emplace_back([&inputs, &results, i]()
                                       {
                auto t = inputs[i].clone();
                results[i] = t * 2 + 1; });
        }
        for (auto &t : batch_threads)
        {
            if (t.joinable())
                t.join();
        }
        return results;
    }

    bool pin_workers(const std::vector<int> &cpus)
    {
#ifdef __linux__
        if (cpus.empty() || workers.empty())
            return false;
        bool ok = true;
        for (size_t i = 0; i < workers.size(); ++i)
        {
            int cpu = cpus[i % cpus.size()];
            if (cpu < 0)
            {
                ok = false;
                continue;
            }
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            int rc = pthread_setaffinity_np(workers[i].native_handle(), sizeof(set), &set);
            if (rc != 0)
                ok = false;
        }
        return ok;
#else
        (void)cpus;
        return false;
#endif
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

void init_thread_manager(py::module &m)
{
    py::class_<ThreadManager>(m, "ThreadManager")
        .def(py::init<int>())
        .def("num_workers", &ThreadManager::num_workers)
        .def("add_task", &ThreadManager::add_task)
        .def("parallel_process", &ThreadManager::parallel_process)
        .def("pin_workers", &ThreadManager::pin_workers);

    m.def("set_process_affinity", [](std::vector<int> cpus)
          { return set_process_affinity_impl(cpus); });
    m.def("get_process_affinity", []()
          { return get_process_affinity_impl(); });
}
