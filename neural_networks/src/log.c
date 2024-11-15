#include "log.h"

const char *level_strings[] = {
    "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"
};

log_level_t global_log_level = LOG_INFO;

void log_log(log_level_t level, const char *file, int line, const char *fmt, ...) {
    if (level < global_log_level) return;

    time_t t = time(NULL);
    struct tm *lt = localtime(&t);
    va_list args;

    char buf[16];
    buf[strftime(buf, sizeof(buf), "%H:%M:%S", lt)] = '\0';

    fprintf(stderr, "%s %-5s %s:%d: ", buf, level_strings[level], file, line);

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);
}

void log_set_level(log_level_t level) {
    global_log_level = level;
}
