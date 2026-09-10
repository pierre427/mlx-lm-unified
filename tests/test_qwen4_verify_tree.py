"""CPU-only unit tests for the shallow-wide MTP verify tree.

Pure topology / mask / positions / acceptance only -- no model load, no Metal.
Everything here runs on synthetic numpy arrays through the pure half of
``qwen4_verify_tree``.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

# Load the module directly from its file. Going through the mlx_lm package
# would run mlx_lm/__init__.py, which imports a modern mlx; the pure half under
# test needs only numpy (mlx is lazy-imported inside the model-facing wrapper),
# so a by-path load keeps this test genuinely CPU-only and mlx-independent.
_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "mlx_lm", "models", "qwen4_verify_tree.py"
)
_spec = importlib.util.spec_from_file_location("qwen4_verify_tree", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod  # dataclass field resolution needs this on 3.9
_spec.loader.exec_module(_mod)

accept_tree = _mod.accept_tree
assemble_mtp_tree = _mod.assemble_mtp_tree
tree_control_entries = _mod.tree_control_entries


def _one_hot_logits(num_nodes, vocab, argmax_per_node):
    """Logits [num_nodes, vocab] whose per-row argmax is argmax_per_node[k]."""
    logits = np.zeros((num_nodes, vocab), dtype=np.float32)
    for k, tok in enumerate(argmax_per_node):
        logits[k, tok] = 10.0
    return logits


class TestTopology(unittest.TestCase):
    def test_depth1_breadth2_order_parent_pos(self):
        t = assemble_mtp_tree(
            root_token=100, root_pos=7, breadth=2, depth=1,
            depth1_tokens=[11, 12],
        )
        self.assertEqual(t.num_nodes, 3)                 # root + 2
        self.assertEqual(t.tokens, (100, 11, 12))
        self.assertEqual(t.parent, (-1, 0, 0))
        self.assertEqual(t.depth, (0, 1, 1))
        self.assertEqual(t.positions, (7, 8, 8))         # siblings share pos
        self.assertEqual(t.children, ((1, 2), (), ()))

    def test_depth1_breadth4_order_parent_pos(self):
        t = assemble_mtp_tree(
            root_token=5, root_pos=0, breadth=4, depth=1,
            depth1_tokens=[21, 22, 23, 24],
        )
        self.assertEqual(t.num_nodes, 5)
        self.assertEqual(t.tokens, (5, 21, 22, 23, 24))
        self.assertEqual(t.parent, (-1, 0, 0, 0, 0))
        self.assertEqual(t.positions, (0, 1, 1, 1, 1))
        self.assertEqual(t.children, ((1, 2, 3, 4), (), (), (), ()))

    def test_depth2_breadth2_order_parent_pos(self):
        t = assemble_mtp_tree(
            root_token=100, root_pos=3, breadth=2, depth=2,
            depth1_tokens=[11, 12],
            depth2_tokens=[[31, 32], [41, 42]],
        )
        # BFS: root, 2 depth-1, then grandchildren grouped by parent in order.
        self.assertEqual(t.num_nodes, 7)                 # 1 + 2 + 2*2
        self.assertEqual(t.tokens, (100, 11, 12, 31, 32, 41, 42))
        self.assertEqual(t.parent, (-1, 0, 0, 1, 1, 2, 2))
        self.assertEqual(t.depth, (0, 1, 1, 2, 2, 2, 2))
        self.assertEqual(t.positions, (3, 4, 4, 5, 5, 5, 5))
        self.assertEqual(
            t.children, ((1, 2), (3, 4), (5, 6), (), (), (), ())
        )

    def test_validation(self):
        with self.assertRaises(ValueError):
            assemble_mtp_tree(root_token=0, root_pos=0, breadth=2, depth=1,
                              depth1_tokens=[1])            # wrong count
        with self.assertRaises(ValueError):
            assemble_mtp_tree(root_token=0, root_pos=0, breadth=2, depth=3,
                              depth1_tokens=[1, 2])         # bad depth
        with self.assertRaises(ValueError):
            assemble_mtp_tree(root_token=0, root_pos=0, breadth=2, depth=2,
                              depth1_tokens=[1, 2])         # missing depth2


class TestTreeMask(unittest.TestCase):
    def _brute_ancestor_mask(self, t):
        n = t.num_nodes
        m = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for a in t.ancestors(i):
                m[i, a] = True
        return m

    def test_mask_is_exactly_ancestor_causal_depth1(self):
        t = assemble_mtp_tree(root_token=100, root_pos=7, breadth=4, depth=1,
                              depth1_tokens=[21, 22, 23, 24])
        np.testing.assert_array_equal(t.tree_mask, self._brute_ancestor_mask(t))
        # Root attends only itself; a sibling never attends another sibling.
        self.assertTrue(t.tree_mask[0, 0])
        self.assertEqual(t.tree_mask[0].sum(), 1)
        self.assertFalse(t.tree_mask[1, 2])
        self.assertTrue(t.tree_mask[1, 0] and t.tree_mask[1, 1])

    def test_mask_is_exactly_ancestor_causal_depth2(self):
        t = assemble_mtp_tree(root_token=100, root_pos=3, breadth=2, depth=2,
                              depth1_tokens=[11, 12],
                              depth2_tokens=[[31, 32], [41, 42]])
        np.testing.assert_array_equal(t.tree_mask, self._brute_ancestor_mask(t))
        # Grandchild node 5 (parent 2, token 41): attends {5, 2, 0} only.
        row5 = np.where(t.tree_mask[5])[0].tolist()
        self.assertEqual(row5, [0, 2, 5])
        # It must NOT see the other branch (nodes 1,3,4) nor its sibling 6.
        for j in (1, 3, 4, 6):
            self.assertFalse(t.tree_mask[5, j])


class TestAccept(unittest.TestCase):
    def test_full_accept_depth2(self):
        t = assemble_mtp_tree(root_token=100, root_pos=3, breadth=2, depth=2,
                              depth1_tokens=[11, 12],
                              depth2_tokens=[[31, 32], [41, 42]])
        # Target greedily wants: root->11 (node1), node1->32 (node4).
        # node4 is a leaf, so acceptance stops there (bonus lives beyond).
        #        node:   0   1   2   3   4   5   6
        argmax = [11, 32, 0, 0, 77, 0, 0]  # node4 (token 32) argmax = 77
        logits = _one_hot_logits(t.num_nodes, 200, argmax)
        accepted, num, seed = accept_tree(t, logits)
        self.assertEqual(accepted, (11, 32))
        self.assertEqual(num, 2)
        self.assertEqual(seed, 4)                    # deepest accepted node
        # Bonus/correction token the caller commits next:
        self.assertEqual(int(np.argmax(logits[seed])), 77)

    def test_all_miss(self):
        t = assemble_mtp_tree(root_token=100, root_pos=3, breadth=2, depth=1,
                              depth1_tokens=[11, 12])
        argmax = [55, 0, 0]                          # root wants 55, not 11/12
        logits = _one_hot_logits(t.num_nodes, 200, argmax)
        accepted, num, seed = accept_tree(t, logits)
        self.assertEqual(accepted, ())
        self.assertEqual(num, 0)
        self.assertEqual(seed, 0)                    # reseed from the root
        self.assertEqual(int(np.argmax(logits[seed])), 55)

    def test_partial_accept_picks_correct_branch(self):
        t = assemble_mtp_tree(root_token=100, root_pos=3, breadth=2, depth=2,
                              depth1_tokens=[11, 12],
                              depth2_tokens=[[31, 32], [41, 42]])
        # root wants 12 (node2, the SECOND child), node2 wants 99 (no child).
        argmax = [12, 0, 99, 0, 0, 0, 0]
        logits = _one_hot_logits(t.num_nodes, 200, argmax)
        accepted, num, seed = accept_tree(t, logits)
        self.assertEqual(accepted, (12,))
        self.assertEqual(num, 1)
        self.assertEqual(seed, 2)
        self.assertEqual(int(np.argmax(logits[seed])), 99)

    def test_accept_one_then_leaf_depth1(self):
        t = assemble_mtp_tree(root_token=100, root_pos=3, breadth=3, depth=1,
                              depth1_tokens=[11, 12, 13])
        argmax = [13, 0, 0, 88]                       # root->13 (node3, a leaf)
        logits = _one_hot_logits(t.num_nodes, 200, argmax)
        accepted, num, seed = accept_tree(t, logits)
        self.assertEqual(accepted, (13,))
        self.assertEqual(num, 1)
        self.assertEqual(seed, 3)
        self.assertEqual(int(np.argmax(logits[seed])), 88)

    def test_argmax_ties_lowest_index(self):
        # Two tokens tie at the root; argmax must pick the lower id (matches
        # mx.argmax) and only a child holding THAT id is accepted.
        t = assemble_mtp_tree(root_token=100, root_pos=0, breadth=2, depth=1,
                              depth1_tokens=[7, 3])
        logits = np.zeros((t.num_nodes, 20), dtype=np.float32)
        logits[0, 3] = logits[0, 7] = 5.0            # tie between ids 3 and 7
        accepted, num, seed = accept_tree(t, logits)
        self.assertEqual(int(np.argmax(logits[0])), 3)
        self.assertEqual(accepted, (3,))             # child token 3 accepted
        self.assertEqual(seed, 2)                    # node index of token 3


class TestControlPositions(unittest.TestCase):
    """The 2026-09-03 tail-block trap: positions advance across depth, and each
    node's tail block is its own (siblings share it, depth advances it)."""

    def test_tail_block_per_node_depth2(self):
        # block_size = IDX_COMPRESS = 4.  root_pos = 3 straddles a boundary.
        t = assemble_mtp_tree(root_token=100, root_pos=3, breadth=2, depth=2,
                              depth1_tokens=[11, 12],
                              depth2_tokens=[[31, 32], [41, 42]])
        entries = tree_control_entries(t, block_size=4)

        # positions: root 3, depth-1 -> 4, depth-2 -> 5.
        self.assertEqual([e["q_pos"] for e in entries], [3, 4, 4, 5, 5, 5, 5])

        # Root at pos 3: length 4 -> 1 closed block, NO open tail (3 % 4... ->
        # length 4 % 4 == 0), tail_block = 3 // 4 = 0.
        root = entries[0]
        self.assertEqual(root["logical_len"], 4)
        self.assertEqual(root["n_blocks"], 1)
        self.assertEqual(root["has_tail"], 0)
        self.assertEqual(root["tail_block"], 0)

        # Depth-1 at pos 4: length 5 -> 1 closed block + open tail; the newest
        # token (its own, pos 4) lives in tail_block 1 and must be attended.
        d1 = entries[1]
        self.assertEqual(d1["logical_len"], 5)
        self.assertEqual(d1["n_blocks"], 1)
        self.assertEqual(d1["has_tail"], 1)
        self.assertEqual(d1["tail_block"], 1)
        self.assertEqual(entries[2]["tail_block"], 1)     # sibling shares it

        # Depth-2 at pos 5: length 6 -> 1 closed block + open tail block 1.
        d2 = entries[3]
        self.assertEqual(d2["logical_len"], 6)
        self.assertEqual(d2["n_blocks"], 1)
        self.assertEqual(d2["has_tail"], 1)
        self.assertEqual(d2["tail_block"], 1)

        # The linear "position + m" rule would put node m at pos root_pos+m
        # (3,4,5,6,7,8,9) -- wrong for a tree.  Confirm we did NOT do that.
        linear = [3 + m for m in range(t.num_nodes)]
        self.assertNotEqual([e["q_pos"] for e in entries], linear)

    def test_tail_block_advances_cleanly_across_boundary(self):
        # root_pos = 4: closed boundary exactly at root, so the open tail
        # appears only from depth 1 onward.
        t = assemble_mtp_tree(root_token=100, root_pos=4, breadth=2, depth=2,
                              depth1_tokens=[11, 12],
                              depth2_tokens=[[31, 32], [41, 42]])
        e = tree_control_entries(t, block_size=4)
        self.assertEqual(e[0]["tail_block"], 1)   # pos4 -> block 1
        self.assertEqual(e[0]["has_tail"], 1)     # length 5, open tail
        self.assertEqual(e[1]["tail_block"], 1)   # pos5 -> still block 1
        self.assertEqual(e[3]["tail_block"], 1)   # pos6 -> still block 1


if __name__ == "__main__":
    unittest.main(verbosity=2)
