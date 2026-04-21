#pragma once

#include <string>
#include <utility>
#include <vector>

namespace mtc_lower
{

    struct DialectArg
    {
        std::string name;
        std::string kind;
        std::string dtype;
        std::string address_space;
    };

    struct DialectValue
    {
        std::string name;
        std::string op;
        std::string dtype;
        std::vector<std::string> inputs;
        std::vector<std::pair<std::string, std::string>> attrs;
    };

    struct DialectStore
    {
        std::string buffer;
        std::string index;
        std::string value;
        std::string mask;
    };

    struct DialectModule
    {
        std::string kernel_name;
        int block_size = 0;
        std::vector<DialectArg> args;
        std::vector<DialectValue> values;
        std::vector<DialectStore> stores;
    };

    std::string find_attr(const DialectValue &value, const std::string &key);

} // namespace mtc_lower
