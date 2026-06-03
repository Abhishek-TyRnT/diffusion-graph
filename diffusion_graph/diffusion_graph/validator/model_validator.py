from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.fx as fx
from torch.fx import Graph, GraphModule, Node
import traceback

@dataclass
class TensorContract:
    op_name: str          # stable key, e.g. "matmul_0"
    # input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]
    # input_dtypes: list[str]
    output_dtypes: list[str]
    location: str         # MLIR location string for error reporting

@dataclass  
class ViolationReport:
    fx_node_name: str
    mlir_op_name: str
    mlir_location: str
    kind: str            # "shape" | "dtype" | "rank"
    expected: object
    actual: object
    input_or_output: str # "input" | "output"
    index: int

ContractMap = dict[str, TensorContract]

def extract_contracts(bytecode: dict) -> ContractMap:
    """
    Walk the diffusion graph bytecode to extract shape and dtype contracts.
    """
    contracts: ContractMap = {}
    
    def shaped_type_to_parts(node: dict) -> tuple[tuple[int, ...], str]:
        if node['diffusion_graph_type'] == 'diffusion_graph.vtensor':
            return tuple(node['shape']), node['dtype']
        elif node['diffusion_graph_type'] == 'scalar':
            return (), node['dtype']
        
        return None

    def _mlir_type_to_dtype_str(dtype: str) -> str:
        mapping = {
            "f32": "torch.float32",
            "f16": "torch.float16",
            "bf16": "torch.bfloat16",
            "f64": "torch.float64",
            "i32": "torch.int32",
            "i64": "torch.int64",
            "i8":  "torch.int8",
            "i1":  "torch.bool",
        }
        return mapping.get(dtype, None)

    op_counters: dict[str, int] = {}

    #Currently do only for main graph only
    bytecode = bytecode['main']

    for ssa_id, op_info in bytecode.items():
        if ssa_id in ['results',
                'entrypoint']:
                    continue
        
        _visit_op(ssa_id, op_info, op_counters, contracts, shaped_type_to_parts)
        
    

    return contracts


def _visit_op(ssa_id, op_info, counters, contracts, shaped_type_to_parts):
    op_type = op_info['op_name']
    counters[op_type] = counters.get(op_type, 0)
    key = f"{op_type}_{counters[op_type]}"
    counters[op_type] += 1

    try:
        in_shapes, in_dtypes, out_shapes, out_dtypes = [], [], [], []

        # for operand in op.operation.operands:
        #     if ir.ShapedType.isinstance(operand.type):
        #         shape, dtype = shaped_type_to_parts(operand.type)
        #         in_shapes.append(shape)
        #         in_dtypes.append(dtype)

        shape, dtype = shaped_type_to_parts(op_info)
        out_shapes.append(shape)
        out_dtypes.append(dtype)

        loc = ssa_id
        contracts[key] = TensorContract(
            op_name=key,
            input_dtypes=in_dtypes,
            output_dtypes=out_dtypes,
            location=loc,
        )
    except Exception:
        pass  # Non-tensor ops, skip




class ShapeViolation(Exception):
    pass

class DtypeViolation(Exception):
    pass



def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype)  # e.g. "torch.float32"

def _check_tensor(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    contract: TensorContract,
    fx_node_name: str,
    io_label: str,
    idx: int,
    violations: list[ViolationReport],
    strict_dynamic: bool = False,
) -> None:
    actual_shape = tuple(tensor.shape)
    actual_dtype = _dtype_str(tensor)

    # Rank check first
    if len(actual_shape) != len(expected_shape):
        violations.append(ViolationReport(
            fx_node_name=fx_node_name,
            mlir_op_name=contract.op_name,
            mlir_location=contract.location,
            kind="rank",
            expected=expected_shape,
            actual=actual_shape,
            input_or_output=io_label,
            index=idx,
        ))
        return  # Shape comparison would be meaningless

    # Per-dim shape check (-1 means dynamic, skip unless strict)
    for dim_i, (exp_dim, act_dim) in enumerate(zip(expected_shape, actual_shape)):
        if exp_dim == -1 and not strict_dynamic:
            continue
        if exp_dim != act_dim:
            violations.append(ViolationReport(
                fx_node_name=fx_node_name,
                mlir_op_name=contract.op_name,
                mlir_location=contract.location,
                kind="shape",
                expected=expected_shape,
                actual=actual_shape,
                input_or_output=io_label,
                index=idx,
            ))
            break

    # Dtype check
    if actual_dtype != expected_dtype:
        violations.append(ViolationReport(
            fx_node_name=fx_node_name,
            mlir_op_name=contract.op_name,
            mlir_location=contract.location,
            kind="dtype",
            expected=expected_dtype,
            actual=actual_dtype,
            input_or_output=io_label,
            index=idx,
        ))

def instrument_graph(
    gm: GraphModule,
    contracts: ContractMap,
    violations: list[ViolationReport],
    strict_dynamic: bool = False,
) -> GraphModule:
    """
    For each fx node that has a matching MLIR contract, insert
    a checker call node immediately after it. Checker nodes are
    tagged with a special meta key so they can be stripped later.
    """
    graph: Graph = gm.graph
    node_list = list(graph.nodes)

    # Build a name->contract mapping using the same counter logic
    # as MLIR extraction. This requires your lowering to preserve
    # op ordering and naming — see note below on stable keys.
    name_to_contract: dict[str, TensorContract] = {}
    counter: dict[str, int] = {}
    for node in node_list:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        base = _fx_node_base_name(node)
        counter[base] = counter.get(base, 0)
        mlir_key = f"{base}_{counter[base]}"
        counter[base] += 1
        if mlir_key in contracts:
            name_to_contract[node.name] = contracts[mlir_key]

    # Inject checker nodes
    for node in node_list:
        if node.name not in name_to_contract:
            continue
        contract = name_to_contract[node.name]

        with graph.inserting_after(node):
            checker_node = graph.call_function(
                _runtime_checker,
                args=(node, contract, node.name, violations, strict_dynamic),
            )
            checker_node.meta["is_contract_checker"] = True
            # Replace all uses of `node` (except the checker itself) 
            # with the checker's output so the chain is: op -> check -> consumers
            node.replace_all_uses_with(checker_node)
            # But the checker needs the original node as input, fix that:
            checker_node.args = (node, contract, node.name, violations, strict_dynamic)

    graph.lint()
    gm.recompile()
    return gm


def _fx_node_base_name(node: Node) -> str:
    """Map an fx node to the same base name used in MLIR op naming."""
    if node.op == "call_function":
        return getattr(node.target, "__name__", str(node.target))
    elif node.op == "call_module":
        return node.target  # module path string
    elif node.op == "call_method":
        return node.target
    return node.name


def _runtime_checker(
    tensor_or_tuple,
    contract: TensorContract,
    fx_node_name: str,
    violations: list[ViolationReport],
    strict_dynamic: bool,
):
    """
    Runs at dummy-forward time. Validates outputs against contract,
    then passes the tensor through unchanged.
    """
    outputs = tensor_or_tuple if isinstance(tensor_or_tuple, (list, tuple)) else [tensor_or_tuple]

    for idx, (tensor, exp_shape, exp_dtype) in enumerate(
        zip(outputs, contract.output_shapes, contract.output_dtypes)
    ):
        if not isinstance(tensor, torch.Tensor):
            continue
        _check_tensor(
            tensor, exp_shape, exp_dtype,
            contract, fx_node_name,
            "output", idx, violations, strict_dynamic,
        )

    # Return original value unmodified
    return tensor_or_tuple

import logging
from typing import Any

log = logging.getLogger(__name__)


class ContractValidator:
    """
    Orchestrates the dummy run, collects violations,
    reports them, and strips checkers for production.
    """

    def __init__(
        self,
        gm: GraphModule,
        contracts: ContractMap,
        strict_dynamic: bool = False,
        raise_on_violation: bool = False,
    ):
        self.original_gm = gm
        self.contracts = contracts
        self.strict_dynamic = strict_dynamic
        self.raise_on_violation = raise_on_violation
        self.violations: list[ViolationReport] = []
        self._validated = False

    def run_validation(self, dummy_inputs: dict[str, Any]) -> list[ViolationReport]:
        """
        Instrument graph, run dummy forward, collect violations.
        Does NOT modify the original graph module.
        """
        import copy
        instrumented_gm = copy.deepcopy(self.original_gm)
        instrument_graph(
            instrumented_gm, 
            self.contracts, 
            self.violations,
            self.strict_dynamic,
        )

        try:
            with torch.no_grad():
                instrumented_gm(**dummy_inputs)
        except Exception as e:
            log.error(f"Dummy run crashed before all checks completed: {e}")
            log.debug(traceback.format_exc())

        self._validated = True
        self._report_violations()

        if self.raise_on_violation and self.violations:
            summary = self._format_violations()
            raise RuntimeError(
                f"MLIR contract validation failed with "
                f"{len(self.violations)} violation(s):\n{summary}"
            )

        return self.violations

    def get_production_graph(self) -> GraphModule:
        """
        Returns the original, unmodified graph ready for production.
        Only call after run_validation() — enforced by assertion.
        """
        assert self._validated, (
            "Call run_validation() before get_production_graph(). "
            "The dummy run must complete (even with violations) so you "
            "can make an informed decision about whether to proceed."
        )
        return self.original_gm

    def _report_violations(self):
        if not self.violations:
            log.info("✓ All MLIR contracts satisfied.")
            return

        log.warning(f"✗ {len(self.violations)} contract violation(s) found:\n")
        log.warning(self._format_violations())

    def _format_violations(self) -> str:
        lines = []
        for v in self.violations:
            lines.append(
                f"  [{v.kind.upper()}] fx_node='{v.fx_node_name}' "
                f"mlir_op='{v.mlir_op_name}' ({v.input_or_output}[{v.index}])\n"
                f"    expected={v.expected}  actual={v.actual}\n"
                f"    @ {v.mlir_location}"
            )
        return "\n".join(lines)

def build_validated_engine(
    mlir_module: ir.Module,
    fx_graph_module: GraphModule,
    user_dummy_inputs: dict | None = None,
    strict_dynamic: bool = False,
    raise_on_violation: bool = True,
    dynamic_dim_size: int = 4,
) -> GraphModule:
    """
    Full pipeline: extract → instrument → validate → strip → return.
    """
    # 1. Extract contracts from the MLIR module
    contracts = extract_contracts(mlir_module)
    log.info(f"Extracted {len(contracts)} op contracts from MLIR module.")

    # 2. Build or accept dummy inputs
    if user_dummy_inputs is None:
        input_contracts = [c for k, c in contracts.items() if k.endswith("_0")]
        dummy_inputs = make_dummy_inputs(
            fx_graph_module, input_contracts, dynamic_dim_size
        )
    else:
        dummy_inputs = user_dummy_inputs

    # 3. Validate (runs instrumented copy, original untouched)
    validator = ContractValidator(
        fx_graph_module, contracts,
        strict_dynamic=strict_dynamic,
        raise_on_violation=raise_on_violation,
    )
    violations = validator.run_validation(dummy_inputs)

    # 4. Return clean production graph
    production_gm = validator.get_production_graph()
    log.info("Returning clean production graph (no checker overhead).")
    return production_gm, violations