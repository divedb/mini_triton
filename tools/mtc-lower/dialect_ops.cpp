#include "dialect_ops.h"

#include <algorithm>

namespace mtc_lower
{

    const std::vector<ValueOpSpec> &supported_value_op_specs()
    {
        static const std::vector<ValueOpSpec> kSpecs = {
            {"program_id", 0, 0, {"axis", "scope"}},
            {"arange", 0, 0, {"start", "end", "step", "scope"}},
            {"cmp_lt", 2, 2, {}},
            {"load", 2, 3, {}},
            {"add", 2, 2, {}},
            {"mul", 2, 2, {}},
        };
        return kSpecs;
    }

    const ValueOpSpec *find_value_op_spec(const std::string &op_name)
    {
        const auto &specs = supported_value_op_specs();
        auto it = std::find_if(specs.begin(), specs.end(), [&](const ValueOpSpec &spec)
                               { return spec.name == op_name; });
        if (it == specs.end())
        {
            return nullptr;
        }
        return &(*it);
    }

    const std::vector<std::string> &supported_value_ops()
    {
        static const std::vector<std::string> kOps = {
            "program_id",
            "arange",
            "cmp_lt",
            "load",
            "add",
            "mul",
        };
        return kOps;
    }

    bool is_supported_value_op(const std::string &op_name)
    {
        return find_value_op_spec(op_name) != nullptr;
    }

} // namespace mtc_lower
