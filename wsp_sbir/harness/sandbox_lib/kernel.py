"""Persistent Python kernel, run inside the sandbox. Protocol: one JSON line in on stdin ({"code": ...}),
one JSON line out on the original stdout ({"out": ...}). Anything the user code prints is captured."""
import ast
import contextlib
import io
import json
import os
import sys
import traceback

proto_out = os.fdopen(os.dup(1), "w")
_sink = os.open("/tmp/kernel_fd.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND)
os.dup2(_sink, 1)  # C-level writes to fd 1/2 must not corrupt the protocol
os.dup2(_sink, 2)
sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(_sink), "wb"), write_through=True)
sys.stderr = sys.stdout

GLOBALS = {"__name__": "__main__"}
with open("/opt/kernel/prelude.py") as f:
    exec(compile(f.read(), "<prelude>", "exec"), GLOBALS)


def run(code):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            if last is not None:
                tree.body = tree.body[:-1]
            exec(compile(tree, "<cell>", "exec"), GLOBALS)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), GLOBALS)
                if value is not None:
                    print(repr(value))
        except BaseException as exc:  # noqa: BLE001 - report everything, keep the kernel alive
            if isinstance(exc, SystemExit):
                print(f"[SystemExit {exc.code}: the kernel keeps running]")
            else:
                tb = traceback.extract_tb(exc.__traceback__)
                tb = [fr for fr in tb if fr.filename != __file__]
                print("Traceback (most recent call last):")
                print("".join(traceback.format_list(tb)), end="")
                print("".join(traceback.format_exception_only(type(exc), exc)), end="")
    return buf.getvalue()


for line in sys.stdin:
    request = json.loads(line)
    proto_out.write(json.dumps({"out": run(request["code"])}) + "\n")
    proto_out.flush()
