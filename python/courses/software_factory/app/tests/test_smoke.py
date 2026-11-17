from helpdesk.smoke import run_smoke


def test_smoke_passes():
    lines: list[str] = []
    assert run_smoke(lines.append), "\n".join(lines)
    assert not any(line.startswith("FAIL") for line in lines)
