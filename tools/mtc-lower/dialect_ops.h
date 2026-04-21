#pragma once

#include <string>
#include <vector>

namespace mtc_lower
{

    struct ValueOpSpec
    {
        std::string name;
        int min_inputs;
        int max_inputs;
        std::vector<std::string> required_attrs;
    };

    const std::vector<std::string> &supported_value_ops();
    bool is_supported_value_op(const std::string &op_name);
    const std::vector<ValueOpSpec> &supported_value_op_specs();
    const ValueOpSpec *find_value_op_spec(const std::string &op_name);

} // namespace mtc_lower
