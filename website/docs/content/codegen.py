import alkahest as ak

pool = ak.ExprPool()
x = pool.symbol("x")
f = ak.sin(x) * ak.exp(x)

# JIT-compile to native code (Cranelift in the default wheel)
fast = ak.compile_expr(f, [x])
print(fast([0.5]))  # 0.7904390832136149

# Or emit dependency-free C99
print(ak.emit_c_expr(f, [x], fn_name="f"))
# double f(double x) {
#     return (exp(x) * sin(x));
# }
