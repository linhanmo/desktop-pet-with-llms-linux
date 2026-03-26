#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <iostream>
#include <condition_variable>
#include <queue>
#include <functional>
namespace py = pybind11;
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
        .def("add_task", &ThreadManager::add_task)
        .def("parallel_process", &ThreadManager::parallel_process);
}
