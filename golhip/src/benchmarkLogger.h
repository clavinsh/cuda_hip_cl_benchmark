#pragma once

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <spdlog/common.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>

class BenchmarkLogger
{
  private:
	std::shared_ptr<spdlog::logger> logger;
	const std::string platform;

  public:
	BenchmarkLogger(const std::string &fileName, const std::string &platform) : platform(platform)
	{
		try
		{
			logger = spdlog::basic_logger_mt("basic_logger", fileName);
			logger->set_pattern("%v");
			logger->info("platform,description,time_ms"); // CSV hederis
		}
		catch (const spdlog::spdlog_ex &ex)
		{
			std::cerr << "Log init failed: " << ex.what() << std::endl;
		}
	}

	void log(const std::string &description, double ms)
	{
		std::stringstream ss;

		ss << platform << "," << description << "," << ms;

		logger->info(ss.str());
	}

	template <typename TimePoint1>
	void chronoLog(const std::string &description, const TimePoint1 &start, const TimePoint1 &end)
	{
		std::chrono::duration<double, std::milli> timeDelta = end - start;
		log(description, timeDelta.count());
	}
};
