#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
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

    std::string read_text_file(const std::string &path)
    {
        std::ifstream stream(path, std::ios::binary);
        if (!stream)
        {
            throw std::runtime_error("failed to read file: " + path);
        }

        std::ostringstream buffer;
        buffer << stream.rdbuf();
        return buffer.str();
    }

    void write_text_file(const std::string &path, const std::string &text)
    {
        std::ofstream stream(path, std::ios::binary);
        if (!stream)
        {
            throw std::runtime_error("failed to write file: " + path);
        }
        stream << text;
    }

    std::string scalar_type(const std::string &dtype)
    {
        if (dtype == "float32")
        {
            return "f32";
        }
        if (dtype == "int32")
        {
            return "i32";
        }
        if (dtype == "index")
        {
            return "i64";
        }
        throw std::runtime_error("unsupported scalar dtype: " + dtype);
    }

    std::string arg_type(const DialectArg &arg)
    {
        if (arg.kind == "buffer")
        {
            return "!llvm.ptr";
        }
        if (arg.kind == "scalar")
        {
            return scalar_type(arg.dtype);
        }
        throw std::runtime_error("unsupported argument kind: " + arg.kind);
    }

    std::string find_attr(const DialectValue &value, const std::string &key)
    {
        for (const auto &pair : value.attrs)
        {
            if (pair.first == key)
            {
                return pair.second;
            }
        }
        throw std::runtime_error("missing required attribute '" + key + "' on " + value.name);
    }

    std::string emit_program_id(const DialectValue &value)
    {
        const std::string axis = find_attr(value, "axis");
        const std::string scope = find_attr(value, "scope");
        if (axis != "0" && axis != "1")
        {
            throw std::runtime_error("unsupported program_id axis: " + axis);
        }
        const std::string axis_suffix = axis == "0" ? "x" : "y";
        if (scope == "'block'" || scope == "block")
        {
            const std::string thread_id_name = value.name + "_thread_id";
            const std::string thread_id_i64_name = thread_id_name + "_i64";
            const std::string zero_name = value.name + "_zero";

            std::ostringstream out;
            out << "%" << thread_id_name << " = nvvm.read.ptx.sreg.tid." << axis_suffix << " : i32\n";
            out << "%" << thread_id_i64_name << " = llvm.sext %" << thread_id_name << " : i32 to i64\n";
            out << "%" << zero_name << " = llvm.mlir.constant(0 : i64) : i64\n";
            out << "%" << value.name << " = llvm.add %" << thread_id_i64_name << ", %" << zero_name << " : i64";
            return out.str();
        }
        if (scope != "'global'" && scope != "global")
        {
            throw std::runtime_error("unsupported program_id scope: " + scope);
        }

        const std::string block_id_name = value.name + "_block_id";
        const std::string block_dim_name = value.name + "_block_dim";
        const std::string thread_id_name = value.name + "_thread_id";
        const std::string block_id_i64_name = block_id_name + "_i64";
        const std::string block_dim_i64_name = block_dim_name + "_i64";
        const std::string thread_id_i64_name = thread_id_name + "_i64";
        const std::string block_base_name = value.name + "_block_base";

        std::ostringstream out;
        out << "%" << block_id_name << " = nvvm.read.ptx.sreg.ctaid." << axis_suffix << " : i32\n";
        out << "%" << block_dim_name << " = nvvm.read.ptx.sreg.ntid." << axis_suffix << " : i32\n";
        out << "%" << thread_id_name << " = nvvm.read.ptx.sreg.tid." << axis_suffix << " : i32\n";
        out << "%" << block_id_i64_name << " = llvm.sext %" << block_id_name << " : i32 to i64\n";
        out << "%" << block_dim_i64_name << " = llvm.sext %" << block_dim_name << " : i32 to i64\n";
        out << "%" << thread_id_i64_name << " = llvm.sext %" << thread_id_name << " : i32 to i64\n";
        out << "%" << block_base_name << " = llvm.mul %" << block_id_i64_name << ", %" << block_dim_i64_name
            << " : i64\n";
        out << "%" << value.name << " = llvm.add %" << block_base_name << ", %" << thread_id_i64_name << " : i64";
        return out.str();
    }

    std::string emit_cmp_lt(const DialectValue &value)
    {
        if (value.inputs.size() != 2)
        {
            throw std::runtime_error("cmp_lt expects exactly 2 inputs");
        }
        return "%" + value.name + " = llvm.icmp \"slt\" %" + value.inputs[0] + ", %" + value.inputs[1] + " : i64";
    }

    std::string emit_load(const DialectValue &value, const std::unordered_map<std::string, DialectArg> &arg_map)
    {
        if (value.inputs.size() < 2)
        {
            throw std::runtime_error("load expects at least 2 inputs");
        }

        const std::string &buffer_name = value.inputs[0];
        const std::string &index_name = value.inputs[1];
        auto found = arg_map.find(buffer_name);
        if (found == arg_map.end())
        {
            throw std::runtime_error("unknown argument referenced in load: " + buffer_name);
        }
        if (found->second.kind != "buffer")
        {
            throw std::runtime_error("load expects a buffer argument: " + buffer_name);
        }

        const std::string ptr_name = value.name + "_ptr";
        const std::string ty = scalar_type(found->second.dtype);

        std::ostringstream out;
        out << "%" << ptr_name << " = llvm.getelementptr %" << buffer_name << "[%" << index_name
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, " << ty << "\n";
        out << "%" << value.name << " = llvm.load %" << ptr_name << " : !llvm.ptr -> " << ty;
        return out.str();
    }

    std::string emit_add(const DialectValue &value)
    {
        if (value.inputs.size() != 2)
        {
            throw std::runtime_error("add expects exactly 2 inputs");
        }

        if (value.dtype == "float32")
        {
            return "%" + value.name + " = llvm.fadd %" + value.inputs[0] + ", %" + value.inputs[1] + " : f32";
        }
        if (value.dtype == "int32")
        {
            return "%" + value.name + " = llvm.add %" + value.inputs[0] + ", %" + value.inputs[1] + " : i32";
        }
        throw std::runtime_error("unsupported add dtype: " + value.dtype);
    }

    std::string emit_mul(const DialectValue &value)
    {
        if (value.inputs.size() != 2)
        {
            throw std::runtime_error("mul expects exactly 2 inputs");
        }

        if (value.dtype == "float32")
        {
            return "%" + value.name + " = llvm.fmul %" + value.inputs[0] + ", %" + value.inputs[1] + " : f32";
        }
        if (value.dtype == "int32")
        {
            return "%" + value.name + " = llvm.mul %" + value.inputs[0] + ", %" + value.inputs[1] + " : i32";
        }
        throw std::runtime_error("unsupported mul dtype: " + value.dtype);
    }

    std::string emit_arange(const DialectValue &value)
    {
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

        const DialectValue base = {
            value.name + "_base",
            "program_id",
            "index",
            {},
            {{"axis", "0"}, {"scope", scope}},
        };

        const std::string start_name = value.name + "_start";
        const std::string step_name = value.name + "_step";
        const std::string stepped_name = value.name + "_stepped";

        std::ostringstream out;
        out << emit_program_id(base) << "\n";
        out << "%" << start_name << " = llvm.mlir.constant(" << start << " : i64) : i64\n";
        if (step == 1)
        {
            out << "%" << value.name << " = llvm.add %" << base.name << ", %" << start_name << " : i64";
            return out.str();
        }

        out << "%" << step_name << " = llvm.mlir.constant(" << step << " : i64) : i64\n";
        out << "%" << stepped_name << " = llvm.mul %" << base.name << ", %" << step_name << " : i64\n";
        out << "%" << value.name << " = llvm.add %" << stepped_name << ", %" << start_name << " : i64";
        return out.str();
    }

    std::string emit_value(const DialectValue &value, const std::unordered_map<std::string, DialectArg> &arg_map)
    {
        if (value.op == "program_id")
        {
            return emit_program_id(value);
        }
        if (value.op == "cmp_lt")
        {
            return emit_cmp_lt(value);
        }
        if (value.op == "load")
        {
            return emit_load(value, arg_map);
        }
        if (value.op == "arange")
        {
            return emit_arange(value);
        }
        if (value.op == "add")
        {
            return emit_add(value);
        }
        if (value.op == "mul")
        {
            return emit_mul(value);
        }
        throw std::runtime_error("unsupported value op: " + value.op);
    }

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

    std::string emit_store(const DialectStore &store, const std::unordered_map<std::string, DialectArg> &arg_map)
    {
        auto found = arg_map.find(store.buffer);
        if (found == arg_map.end())
        {
            throw std::runtime_error("unknown argument referenced in store: " + store.buffer);
        }
        if (found->second.kind != "buffer")
        {
            throw std::runtime_error("store expects a buffer argument: " + store.buffer);
        }

        const std::string ty = scalar_type(found->second.dtype);
        const std::string ptr_name = store.value + "_store_ptr";

        std::ostringstream out;
        out << "%" << ptr_name << " = llvm.getelementptr %" << store.buffer << "[%" << store.index
            << "] : (!llvm.ptr, i64) -> !llvm.ptr, " << ty << "\n";
        out << "llvm.store %" << store.value << ", %" << ptr_name << " : " << ty << ", !llvm.ptr";
        return out.str();
    }

    std::string common_guard_mask(const DialectModule &module)
    {
        std::set<std::string> masks;
        for (const auto &store : module.stores)
        {
            if (!store.mask.empty())
            {
                masks.insert(store.mask);
            }
        }

        if (masks.empty())
        {
            return std::string();
        }
        if (masks.size() != 1)
        {
            throw std::runtime_error("multiple distinct store masks are not supported");
        }

        const std::string mask = *masks.begin();
        for (const auto &value : module.values)
        {
            if (value.op == "load" && value.inputs.size() == 3 && value.inputs[2] != mask)
            {
                throw std::runtime_error("multiple distinct load masks are not supported");
            }
        }
        return mask;
    }

    std::set<std::string> guarded_value_names(const DialectModule &module, const std::string &guard_name)
    {
        std::set<std::string> guarded;
        if (guard_name.empty())
        {
            return guarded;
        }

        for (const auto &value : module.values)
        {
            if (value.op == "load" && value.inputs.size() == 3 && value.inputs[2] == guard_name)
            {
                guarded.insert(value.name);
            }
        }

        bool changed = true;
        while (changed)
        {
            changed = false;
            for (const auto &value : module.values)
            {
                if (guarded.find(value.name) != guarded.end())
                {
                    continue;
                }

                for (const auto &input : value.inputs)
                {
                    if (guarded.find(input) != guarded.end())
                    {
                        guarded.insert(value.name);
                        changed = true;
                        break;
                    }
                }
            }
        }

        return guarded;
    }

    std::string emit_mlir_from_dialect(const DialectModule &module)
    {
        std::unordered_map<std::string, DialectArg> arg_map;
        for (const auto &arg : module.args)
        {
            arg_map[arg.name] = arg;
        }

        std::ostringstream signature;
        for (size_t index = 0; index < module.args.size(); ++index)
        {
            if (index > 0)
            {
                signature << ", ";
            }
            signature << "%" << module.args[index].name << ": " << arg_type(module.args[index]);
        }

        const std::string guard_name = common_guard_mask(module);
        const std::set<std::string> guarded_names = guarded_value_names(module, guard_name);

        std::ostringstream out;
        out << "module {\n";
        out << "  llvm.func @" << module.kernel_name << "(" << signature.str() << ") attributes "
            << "{nvvm.kernel, nvvm.maxntid = array<i32: " << module.block_size << ", 1, 1>, "
            << "mini_triton.block_size = " << module.block_size << " : i32} {\n";

        for (const auto &value : module.values)
        {
            if (guarded_names.find(value.name) != guarded_names.end())
            {
                continue;
            }
            std::istringstream lines(emit_value(value, arg_map));
            std::string line;
            while (std::getline(lines, line))
            {
                out << "    " << line << "\n";
            }
        }

        if (!guard_name.empty())
        {
            out << "    llvm.cond_br %" << guard_name << ", ^bb1, ^bb2\n";
            out << "  ^bb1:\n";

            for (const auto &value : module.values)
            {
                if (guarded_names.find(value.name) == guarded_names.end())
                {
                    continue;
                }
                std::istringstream lines(emit_value(value, arg_map));
                std::string line;
                while (std::getline(lines, line))
                {
                    out << "    " << line << "\n";
                }
            }

            for (const auto &store : module.stores)
            {
                if (store.mask.empty())
                {
                    throw std::runtime_error("mixed masked and unmasked stores are not supported");
                }
                std::istringstream lines(emit_store(store, arg_map));
                std::string line;
                while (std::getline(lines, line))
                {
                    out << "    " << line << "\n";
                }
            }

            out << "    llvm.br ^bb2\n";
            out << "  ^bb2:\n";
        }
        else
        {
            for (const auto &store : module.stores)
            {
                std::istringstream lines(emit_store(store, arg_map));
                std::string line;
                while (std::getline(lines, line))
                {
                    out << "    " << line << "\n";
                }
            }
        }

        out << "    llvm.return\n";
        out << "  }\n";
        out << "}\n";
        return out.str();
    }

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

                if (tokens[4] != "-")
                {
                    value.inputs = split(tokens[4], ',');
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

    bool starts_with(const std::string &value, const std::string &prefix)
    {
        return value.rfind(prefix, 0) == 0;
    }

    std::string quote_arg(const std::string &value)
    {
        if (value.empty())
        {
            return "\"\"";
        }

        bool requires_quotes = false;
        for (char ch : value)
        {
            if (ch == ' ' || ch == '\t' || ch == '"')
            {
                requires_quotes = true;
                break;
            }
        }

        if (!requires_quotes)
        {
            return value;
        }

        std::string out;
        out.reserve(value.size() + 2);
        out.push_back('"');
        for (char ch : value)
        {
            if (ch == '"')
            {
                out.push_back('\\');
            }
            out.push_back(ch);
        }
        out.push_back('"');
        return out;
    }

    std::string join_command(const std::vector<std::string> &argv)
    {
        std::string command;
        for (size_t index = 0; index < argv.size(); ++index)
        {
            if (index > 0)
            {
                command.push_back(' ');
            }
            command += quote_arg(argv[index]);
        }
        return command;
    }

    int run_command(const std::vector<std::string> &argv)
    {
        const std::string command = join_command(argv);
        const int rc = std::system(command.c_str());
        if (rc != 0)
        {
            std::cerr << "mini_triton_lower: command failed (" << rc << "): " << command << '\n';
        }
        return rc;
    }

    void print_usage()
    {
        std::cerr
            << "Usage: mini_triton_lower --mlir-opt <path> --mlir-translate <path> --llc <path> "
            << "[--input-dialect <path>] --input-mlir <path> --optimized-mlir <path> --llvm-ir <path> --ptx <path> "
            << "--cuda-arch <sm_xx>\n";
    }

} // namespace

int main(int argc, char **argv)
{
    std::unordered_map<std::string, std::string> args;
    for (int index = 1; index < argc; ++index)
    {
        std::string key = argv[index];
        if (!starts_with(key, "--"))
        {
            std::cerr << "mini_triton_lower: unexpected positional argument: " << key << '\n';
            print_usage();
            return 2;
        }

        if (index + 1 >= argc)
        {
            std::cerr << "mini_triton_lower: missing value for argument: " << key << '\n';
            print_usage();
            return 2;
        }

        args[key] = argv[++index];
    }

    const std::vector<std::string> required = {
        "--mlir-opt",
        "--mlir-translate",
        "--llc",
        "--input-mlir",
        "--optimized-mlir",
        "--llvm-ir",
        "--ptx",
        "--cuda-arch",
    };

    for (const std::string &key : required)
    {
        if (args.find(key) == args.end())
        {
            std::cerr << "mini_triton_lower: missing required argument: " << key << '\n';
            print_usage();
            return 2;
        }
    }

    try
    {
        auto dialect_it = args.find("--input-dialect");
        if (dialect_it != args.end())
        {
            const std::string dialect_text = read_text_file(dialect_it->second);
            const DialectModule module = parse_dialect_text(dialect_text);
            const std::string generated_mlir = emit_mlir_from_dialect(module);
            write_text_file(args["--input-mlir"], generated_mlir);
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "mini_triton_lower: failed to lower mini_triton dialect: " << ex.what() << '\n';
        return 2;
    }

    const int mlir_opt_rc = run_command({
        args["--mlir-opt"],
        args["--input-mlir"],
        "--convert-nvvm-to-llvm",
        "--reconcile-unrealized-casts",
        "-o",
        args["--optimized-mlir"],
    });
    if (mlir_opt_rc != 0)
    {
        return 1;
    }

    const int mlir_translate_rc = run_command({
        args["--mlir-translate"],
        "--mlir-to-llvmir",
        args["--optimized-mlir"],
        "-o",
        args["--llvm-ir"],
    });
    if (mlir_translate_rc != 0)
    {
        return 1;
    }

    const int llc_rc = run_command({
        args["--llc"],
        "-march=nvptx64",
        std::string("-mcpu=") + args["--cuda-arch"],
        args["--llvm-ir"],
        "-o",
        args["--ptx"],
    });
    if (llc_rc != 0)
    {
        return 1;
    }

    return 0;
}
