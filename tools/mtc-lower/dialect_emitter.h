#pragma once

#include "dialect_model.h"

#include <string>

namespace mtc_lower
{

    std::string emit_mlir_from_dialect(const DialectModule &module);

} // namespace mtc_lower
