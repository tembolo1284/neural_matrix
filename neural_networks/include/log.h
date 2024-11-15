#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>

typedef enum {
    LOG_TRACE,
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_FATAL
} log_level_t;

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define log_trace(...) log_log(LOG_TRACE, __FILENAME__, __LINE__, __VA_ARGS__)
#define log_debug(...) log_log(LOG_DEBUG, __FILENAME__, __LINE__, __VA_ARGS__)
#define log_info(...)  log_log(LOG_INFO,  __FILENAME__, __LINE__, __VA_ARGS__)
#define log_warn(...)  log_log(LOG_WARN,  __FILENAME__, __LINE__, __VA_ARGS__)
#define log_error(...) log_log(LOG_ERROR, __FILENAME__, __LINE__, __VA_ARGS__)
#define log_fatal(...) log_log(LOG_FATAL, __FILENAME__, __LINE__, __VA_ARGS__)

extern const char *level_strings[];
extern log_level_t global_log_level;

void log_log(log_level_t level, const char *file, int line, const char *fmt, ...);
void log_set_level(log_level_t level);

#endif // LOG_H
