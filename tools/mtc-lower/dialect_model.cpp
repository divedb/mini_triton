#include "dialect_model.h"

#include <stdexcept>

namespace mtc_lower
{

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

} // namespace mtc_lower
