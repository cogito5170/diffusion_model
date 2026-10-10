"""The project-level Claude Code subagents exist, are narrowly scoped, and have limited tools."""

import re
import unittest

from tests.helpers import ROOT

REQUIRED = {
    "concept-researcher": {"Read", "Write", "WebSearch", "WebFetch"},
    "concept-generator": {"Read", "Write"},
    "art-director": {"Read", "Write"},
    "material-specialist": {"Read", "Write"},
    "prompt-compiler": {"Read", "Bash"},
    "image-operator": {"Read", "Bash"},
    "art-critic": {"Read", "Write"},
    "curator": {"Read", "Write"},
}
FORBIDDEN = {"Edit", "NotebookEdit", "Agent", "Task"}


def frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert m, "missing frontmatter"
    out = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        out[k.strip()] = v.strip()
    return out, text[m.end():]


class AgentDefinitionTests(unittest.TestCase):
    def test_all_required_agents_defined_with_limited_tools(self):
        for name, allowed in REQUIRED.items():
            with self.subTest(agent=name):
                path = ROOT / ".claude" / "agents" / f"{name}.md"
                self.assertTrue(path.exists())
                fm, body = frontmatter(path.read_text())
                self.assertEqual(fm["name"], name)
                self.assertTrue(fm.get("description"))
                tools = {t.strip() for t in fm["tools"].split(",")}
                self.assertTrue(tools <= allowed, f"{name} has extra tools {tools - allowed}")
                self.assertFalse(tools & FORBIDDEN)
                for heading in ("## Inputs", "## Output", "## Rules"):
                    self.assertIn(heading, body)

    def test_agents_reference_real_schemas(self):
        for name in REQUIRED:
            body = (ROOT / ".claude" / "agents" / f"{name}.md").read_text()
            for ref in re.findall(r"atelier/schemas/([a-z_]+\.v1\.json)", body):
                self.assertTrue((ROOT / "atelier" / "schemas" / ref).exists(), f"{name} cites missing {ref}")


if __name__ == "__main__":
    unittest.main()
