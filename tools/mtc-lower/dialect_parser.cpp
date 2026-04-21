#include "dialect_parser.h"

#include "dialect_ops.h"
#include "dialect_verifier.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtc_lower
{

    namespace
    {

        std::vector<std::string> split(const std::string &value, char delimiter)
        {
            std::vector<std::string> out;
            std::string current;
            std::istringstream stream(value);
            while (std::getline(stream, current, delimiter))
            {
                out.push_back(current);
            }
            return out;
        }

        std::vector<std::string> split_whitespace(const std::string &line)
        {
            std::vector<std::string> out;
            std::istringstream stream(line);
            std::string token;
            while (stream >> token)
            {
                out.push_back(token);
            }
            return out;
        }

    } // namespace

    DialectModule parse_dialect_text(const std::string &text)
    {
        DialectModule module;
        std::istringstream stream(text);
        std::string line;
        int line_number = 0;

        while (std::getline(stream, line))
        {
            ++line_number;
            if (line.empty())
            {
                continue;
            }

            const std::vector<std::string> tokens = split_whitespace(line);
            if (tokens.empty())
            {
                continue;
            }

            if (tokens[0] == "kernel")
            {
                if (tokens.size() != 3)
                {
                    throw std::runtime_error("invalid kernel record at line " + std::to_string(line_number));
                }
                module.kernel_name = tokens[1];
                module.block_size = std::stoi(tokens[2]);
                continue;
            }

            if (tokens[0] == "arg")
            {
                if (tokens.size() != 5)
                {
                    throw std::runtime_error("invalid arg record at line " + std::to_string(line_number));
                }

                DialectArg arg;
                arg.name = tokens[1];
                arg.kind = tokens[2];
                arg.dtype = tokens[3];
                arg.address_space = tokens[4];
                module.args.push_back(arg);
                continue;
            }

            if (tokens[0] == "value")
            {
                if (tokens.size() != 6)
                {
                    throw std::runtime_error("invalid value record at line " + std::to_string(line_number));
                }

                DialectValue value;
                value.name = tokens[1];
                value.op = tokens[2];
                value.dtype = tokens[3];

                if (!is_supported_value_op(value.op))
                {
                    throw std::runtime_error(
                        "unsupported value op at line " + std::to_string(line_number) + ": " + value.op);
                }

                if (tokens[4] != "-")
                {
                    value.inputs = split(tokens[4], ',');
                }

                const ValueOpSpec *op_spec = find_value_op_spec(value.op);
                if (op_spec == nullptr)
                {
                    throw std::runtime_error(
                        "missing value op spec at line " + std::to_string(line_number) + ": " + value.op);
                }
                if (static_cast<int>(value.inputs.size()) < op_spec->min_inputs ||
                    static_cast<int>(value.inputs.size()) > op_spec->max_inputs)
                {
                    throw std::runtime_error(
                        "invalid input count for value op '" + value.op + "' at line " +
                        std::to_string(line_number) + ": got " + std::to_string(value.inputs.size()) +
                        ", expected " + std::to_string(op_spec->min_inputs) +
                        (op_spec->min_inputs == op_spec->max_inputs
                             ? std::string()
                             : ".." + std::to_string(op_spec->max_inputs)));
                }

                if (tokens[5] != "-")
                {
                    const std::vector<std::string> attrs = split(tokens[5], ',');
                    for (const auto &attr : attrs)
                    {
                        const size_t eq = attr.find('=');
                        if (eq == std::string::npos)
                        {
                            throw std::runtime_error("invalid attribute at line " + std::to_string(line_number));
                        }
                        value.attrs.push_back({attr.substr(0, eq), attr.substr(eq + 1)});
                    }
                }

                for (const auto &required_attr : op_spec->required_attrs)
                {
                    const bool present = std::any_of(
                        value.attrs.begin(), value.attrs.end(), [&](const std::pair<std::string, std::string> &item)
                        { return item.first == required_attr; });
                    if (!present)
                    {
                        throw std::runtime_error(
                            "missing required attribute '" + required_attr + "' for value op '" + value.op +
                            "' at line " + std::to_string(line_number));
                    }
                }

                for (const auto &parsed_attr : value.attrs)
                {
                    const bool expected = std::find(
                                              op_spec->required_attrs.begin(),
                                              op_spec->required_attrs.end(),
                                              parsed_attr.first) != op_spec->required_attrs.end();
                    if (!expected && !op_spec->required_attrs.empty())
                    {
                        throw std::runtime_error(
                            "unexpected attribute '" + parsed_attr.first + "' for value op '" + value.op +
                            "' at line " + std::to_string(line_number));
                    }
                }

                module.values.push_back(value);
                continue;
            }

            if (tokens[0] == "store")
            {
                if (tokens.size() != 5)
                {
                    throw std::runtime_error("invalid store record at line " + std::to_string(line_number));
                }

                DialectStore store;
                store.buffer = tokens[1];
                store.index = tokens[2];
                store.value = tokens[3];
                store.mask = tokens[4] == "-" ? "" : tokens[4];
                module.stores.push_back(store);
                continue;
            }

            throw std::runtime_error("unknown record type at line " + std::to_string(line_number) + ": " + tokens[0]);
        }

        if (module.kernel_name.empty())
        {
            throw std::runtime_error("dialect text does not contain a kernel record");
        }
        if (module.block_size <= 0)
        {
            throw std::runtime_error("dialect kernel block_size must be positive");
        }

        verify_dialect_module(module);
        return module;
    }

} // namespace mtc_lower
