include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

#ifndef MINI_TRITON_OPS
#define MINI_TRITON_OPS

def MiniTriton_Dialect : Dialect {
  let name = "mini_triton";
  let cppNamespace = "::mini_triton";
  let summary = "Mini Triton bootstrap dialect";
  let description = [{
    Declarative operation definitions for the mini_triton IR surface used by
    the C++ lowering driver. This file is currently a schema source and is not
    yet wired into generated op classes.
  }];
}

class MiniTriton_Op<string mnemonic, list<Trait> traits = []> :
    Op<MiniTriton_Dialect, mnemonic, traits>;

def MiniTriton_ProgramIdOp : MiniTriton_Op<"program_id"> {
  let summary = "Program index primitive";
  let arguments = (ins I64Attr:$axis, StrAttr:$scope);
  let results = (outs Index:$result);
}

def MiniTriton_ArangeOp : MiniTriton_Op<"arange"> {
  let summary = "Range index primitive";
  let arguments = (ins I64Attr:$start, I64Attr:$end, I64Attr:$step, StrAttr:$scope);
  let results = (outs Index:$result);
}

def MiniTriton_CmpLtOp : MiniTriton_Op<"cmp_lt"> {
  let summary = "Signed less-than comparison";
  let arguments = (ins Index:$lhs, Index:$rhs);
  let results = (outs I1:$result);
}

def MiniTriton_LoadOp : MiniTriton_Op<"load"> {
  let summary = "Masked or unmasked buffer load";
  let arguments = (ins AnyType:$buffer, Index:$index, Optional<I1>:$mask);
  let results = (outs AnyType:$result);
}

def MiniTriton_StoreOp : MiniTriton_Op<"store"> {
  let summary = "Masked or unmasked buffer store";
  let arguments = (ins AnyType:$buffer, Index:$index, AnyType:$value, Optional<I1>:$mask);
}

def MiniTriton_AddOp : MiniTriton_Op<"add"> {
  let summary = "Elementwise add";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def MiniTriton_MulOp : MiniTriton_Op<"mul"> {
  let summary = "Elementwise multiply";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

#endif // MINI_TRITON_OPS
