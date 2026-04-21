#include "dialect_verifier.h"

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mtc_lower
{

    namespace
    {

        std::string find_value_dtype_or_throw(
            const std::string &name,
            const std::unordered_map<std::string, DialectArg> &arg_map,
            const std::unordered_map<std::string, std::string> &value_types)
        {
            auto arg_it = arg_map.find(name);
            if (arg_it != arg_map.end())
            {
                return arg_it->second.dtype;
            }

            auto value_it = value_types.find(name);
            if (value_it != value_types.end())
            {
                return value_it->second;
            }

            throw std::runtime_error("unknown symbol referenced: " + name);
        }

        void verify_program_id(const DialectValue &value)
        {
            if (value.dtype != "index")
            {
                throw std::runtime_error("program_id must produce index dtype");
            }
            if (!value.inputs.empty())
            {
                throw std::runtime_error("program_id does not accept inputs");
            }

            const std::string axis = find_attr(value, "axis");
            const std::string scope = find_attr(value, "scope");
            if (axis != "0" && axis != "1")
            {
                throw std::runtime_error("unsupported program_id axis: " + axis);
            }
            if (scope != "global" && scope != "'global'" && scope != "block" && scope != "'block'")
            {
                throw std::runtime_error("unsupported program_id scope: " + scope);
            }
        }

        void verify_arange(const DialectValue &value)
        {
            if (value.dtype != "index")
            {
                throw std::runtime_error("arange must produce index dtype");
            }
            if (!value.inputs.empty())
            {
                throw std::runtime_error("arange does not accept inputs");
            }

            const int start = std::stoi(find_attr(value, "start"));
            const int end = std::stoi(find_attr(value, "end"));
            const int step = std::stoi(find_attr(value, "step"));
            const std::string scope = find_attr(value, "scope");
            if (end <= start)
            {
                throw std::runtime_error(
                    "invalid arange bounds: start=" + std::to_string(start) + ", end=" + std::to_string(end));
            }
            if (step <= 0)
            {
                throw std::runtime_error("invalid arange step: step=" + std::to_string(step));
            }
            if (scope != "global" && scope != "'global'" && scope != "block" && scope != "'block'")
            {
                throw std::runtime_error("unsupported arange scope: " + scope);
            }
        }

        void verify_cmp_lt(
            const DialectValue &value,
            const std::unordered_map<std::string, DialectArg> &arg_map,
            const std::unordered_map<std::string, std::string> &value_types)
        {
            if (value.inputs.size() != 2)
            {
                throw std::runtime_error("cmp_lt expects exactly 2 inputs");
            }
            if (value.dtype != "pred")
            {
                throw std::runtime_error("cmp_lt must produce pred dtype");
            }

            const std::string lhs_type = find_value_dtype_or_throw(value.inputs[0], arg_map, value_types);
            const std::string rhs_type = find_value_dtype_or_throw(value.inputs[1], arg_map, value_types);
            if (lhs_type != rhs_type)
            {
                throw std::runtime_error("cmp_lt input dtypes must match");
            }
            if (lhs_type != "index")
            {
                throw std::runtime_error("cmp_lt currently supports index operands only");
            }
        }

        void verify_load(
            const DialectValue &value,
            const std::unordered_map<std::string, DialectArg> &arg_map,
            const std::unordered_map<std::string, std::string> &value_types)
        {
            if (value.inputs.size() != 2 && value.inputs.size() != 3)
            {
                throw std::runtime_error("load expects 2 or 3 inputs");
            }

            const auto arg_it = arg_map.find(value.inputs[0]);
            if (arg_it == arg_map.end())
            {
                throw std::runtime_error("load expects first input to be a kernel argument");
            }
            if (arg_it->second.kind != "buffer")
            {
                throw std::runtime_error("load expects first input to be a buffer argument");
            }
            if (arg_it->second.dtype != value.dtype)
            {
                throw std::runtime_error("load result dtype must match buffer dtype");
            }

            const std::string index_type = find_value_dtype_or_throw(value.inputs[1], arg_map, value_types);
            if (index_type != "index")
            {
                throw std::runtime_error("load index must have index dtype");
            }

            if (value.inputs.size() == 3)
            {
                const std::string mask_type = find_value_dtype_or_throw(value.inputs[2], arg_map, value_types);
                if (mask_type != "pred")
                {
                    throw std::runtime_error("load mask must have pred dtype");
                }
            }
        }

        void verify_binary_numeric(
            const DialectValue &value,
            const std::string &op_name,
            const std::unordered_map<std::string, DialectArg> &arg_map,
            const std::unordered_map<std::string, std::string> &value_types)
        {
            if (value.inputs.size() != 2)
            {
                throw std::runtime_error(op_name + " expects exactly 2 inputs");
            }

            if (value.dtype != "float32" && value.dtype != "int32")
            {
                throw std::runtime_error(op_name + " currently supports float32 or int32 only");
            }

            const std::string lhs_type = find_value_dtype_or_throw(value.inputs[0], arg_map, value_types);
            const std::string rhs_type = find_value_dtype_or_throw(value.inputs[1], arg_map, value_types);
            if (lhs_type != value.dtype || rhs_type != value.dtype)
            {
                throw std::runtime_error(op_name + " input dtypes must match result dtype");
            }
        }

    } // namespace

    void verify_dialect_module(const DialectModule &module)
    {
        if (module.kernel_name.empty())
        {
            throw std::runtime_error("dialect module has empty kernel name");
        }
        if (module.block_size <= 0)
        {
            throw std::runtime_error("dialect kernel block_size must be positive");
        }

        std::unordered_map<std::string, DialectArg> arg_map;
        for (const auto &arg : module.args)
        {
            if (arg_map.find(arg.name) != arg_map.end())
            {
                throw std::runtime_error("duplicate kernel argument name: " + arg.name);
            }
            arg_map[arg.name] = arg;
        }

        std::unordered_map<std::string, std::string> value_types;
        for (const auto &value : module.values)
        {
            if (value_types.find(value.name) != value_types.end())
            {
                throw std::runtime_error("duplicate value name: " + value.name);
            }
            if (arg_map.find(value.name) != arg_map.end())
            {
                throw std::runtime_error("value name collides with argument name: " + value.name);
            }

            if (value.op == "program_id")
            {
                verify_program_id(value);
            }
            else if (value.op == "arange")
            {
                verify_arange(value);
            }
            else if (value.op == "cmp_lt")
            {
                verify_cmp_lt(value, arg_map, value_types);
            }
            else if (value.op == "load")
            {
                verify_load(value, arg_map, value_types);
            }
            else if (value.op == "add")
            {
                verify_binary_numeric(value, "add", arg_map, value_types);
            }
            else if (value.op == "mul")
            {
                verify_binary_numeric(value, "mul", arg_map, value_types);
            }
            else
            {
                throw std::runtime_error("unsupported value op: " + value.op);
            }

            value_types[value.name] = value.dtype;
        }

        if (module.stores.empty())
        {
            throw std::runtime_error("dialect module must contain at least one store");
        }

        for (const auto &store : module.stores)
        {
            const auto arg_it = arg_map.find(store.buffer);
            if (arg_it == arg_map.end())
            {
                throw std::runtime_error("store references unknown buffer argument: " + store.buffer);
            }
            if (arg_it->second.kind != "buffer")
            {
                throw std::runtime_error("store target must be a buffer argument: " + store.buffer);
            }

            const std::string index_type = find_value_dtype_or_throw(store.index, arg_map, value_types);
            if (index_type != "index")
            {
                throw std::runtime_error("store index must have index dtype");
            }

            const std::string value_type = find_value_dtype_or_throw(store.value, arg_map, value_types);
            if (value_type != arg_it->second.dtype)
            {
                throw std::runtime_error("store value dtype must match destination buffer dtype");
            }

            if (!store.mask.empty())
            {
                const std::string mask_type = find_value_dtype_or_throw(store.mask, arg_map, value_types);
                if (mask_type != "pred")
                {
                    throw std::runtime_error("store mask must have pred dtype");
                }
            }
        }
    }

} // namespace mtc_lower
