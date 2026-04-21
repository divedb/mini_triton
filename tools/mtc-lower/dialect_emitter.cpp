#include "dialect_emitter.h"

#include "dialect_model.h"

#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace mtc_lower
{

    namespace
    {

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
                throw std::runtime_error("invalid arange bounds: start=" + std::to_string(start) + ", end=" + std::to_string(end));
            }
            if (step <= 0)
            {
                throw std::runtime_error("invalid arange step: step=" + std::to_string(step));
            }

            const DialectValue base = {value.name + "_base", "program_id", "index", {}, {{"axis", "0"}, {"scope", scope}}};

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

    } // namespace

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

} // namespace mtc_lower
