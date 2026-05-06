using HighDimPDE, Aqua
using NNlib: NNlib
@testset "Aqua" begin
    Aqua.find_persistent_tasks_deps(HighDimPDE)
    Aqua.test_ambiguities(HighDimPDE, recursive = false)
    Aqua.test_deps_compat(HighDimPDE)
    # `NNlib.fast_act(::typeof(tanh), ::CuArray) = tanh` is intentional
    # type piracy: a CUDA-side opt-out of NNlib's `tanh_fast` substitution
    # that works around an `InvalidIRError` when broadcasting
    # `ComposedFunction{tanh_fast, +}` on the GPU. See FluxML/Flux.jl#2633
    # for the analogous Metal report and resolution.
    Aqua.test_piracies(HighDimPDE, treat_as_own = [NNlib.fast_act])
    Aqua.test_project_extras(HighDimPDE)
    Aqua.test_stale_deps(HighDimPDE)
    Aqua.test_unbound_args(HighDimPDE)
    Aqua.test_undefined_exports(HighDimPDE)
end
