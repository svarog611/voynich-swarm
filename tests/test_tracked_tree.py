"""What git tracks must work in a fresh checkout on any machine.

On 28 Nov 2026 a symlink named `data`, pointing at a scratch directory on one laptop, was committed
with a code PR (#772). In every other checkout `python3 -m vs fetch` then failed, so the advisory
check and the scorer stopped for every open PR until the link was removed (#797). `.gitignore` had
`data/raw/`, which matches a directory but not a link called `data`.
"""
import os
import subprocess
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
LOCAL_ONLY = ("data", "secrets", "_pr")      # downloaded inputs, the season seed, scorer scratch


def problems(entries):
    """entries: (git mode, path) for every tracked file. Returns one line per thing that must not be tracked."""
    out = []
    for mode, name in entries:
        if mode == "120000":
            out.append(f"{name}: a symlink (its target exists only on the machine that made it)")
        elif mode == "160000":
            out.append(f"{name}: a submodule")
        if name.split("/")[0] in LOCAL_ONLY:
            out.append(f"{name}: {name.split('/')[0]}/ is local to a checkout and is never committed")
    return out


def tracked():
    listing = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [(row.split(" ", 1)[0], row.split("\t", 1)[1]) for row in listing.split("\0") if row]


class TrackedTreeTest(unittest.TestCase):
    def test_the_rule_names_links_submodules_and_local_directories(self):
        self.assertEqual(problems([("100644", "vs/fetch.py"), ("100755", "tools/score_pr.sh"),
                                   ("100644", "experiments/data/x/experiment.json")]), [])
        self.assertEqual(len(problems([("120000", "data")])), 2)
        self.assertEqual(len(problems([("120000", "vs/shortcut")])), 1)
        self.assertEqual(len(problems([("160000", "vendor")])), 1)
        self.assertEqual(len(problems([("100644", "data/raw/ZL3b-n.txt"), ("100644", "secrets/heldout_seed.json")])), 2)

    def test_nothing_tracked_breaks_a_fresh_checkout(self):
        if not os.path.exists(os.path.join(ROOT, ".git")):
            self.skipTest("not a git checkout")
        self.assertEqual(problems(tracked()), [])

    def test_git_ignores_a_link_named_like_a_local_directory(self):
        if not os.path.exists(os.path.join(ROOT, ".git")):
            self.skipTest("not a git checkout")
        for name in LOCAL_ONLY:      # no trailing slash: asks about a file or link of that name, not a directory
            ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", name], cwd=ROOT).returncode == 0
            self.assertTrue(ignored, f".gitignore must ignore /{name} whether it is a directory or a link")


if __name__ == "__main__":
    unittest.main()
