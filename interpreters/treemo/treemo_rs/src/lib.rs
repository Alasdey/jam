use memchr::memmem;
use pyo3::prelude::*;

fn build_parent_list(code: &[u8]) -> Vec<i32> {
    let mut parent = Vec::with_capacity(code.len() + 1);
    parent.push(-1i32); // root sentinel
    let mut current: i32 = 0;
    for &bit in code {
        if bit == 1 {
            parent.push(current);
            current = (parent.len() - 1) as i32;
        } else {
            current = parent[current as usize];
            parent.push(current);
        }
    }
    parent
}

// Replicates greedy_pairs: finds nodes with exactly 4 children, extracts
// (lhs, rhs) code span pairs, consuming matched entries from the parent list.
fn greedy_pairs(mut parent: Vec<i32>) -> Vec<([usize; 2], [usize; 2])> {
    let mut rules = Vec::new();
    let max_node = match parent.iter().filter(|&&x| x >= 0).max() {
        Some(&m) => m as usize,
        None => return rules,
    };
    let mut offset = 0usize;

    for node in 0..max_node {
        let inds: Vec<usize> = parent
            .iter()
            .enumerate()
            .filter(|(_, &x)| x == node as i32)
            .map(|(i, _)| i)
            .collect();

        if inds.len() == 4 {
            let lhs = [offset + inds[0] - 1, offset + inds[1]];
            let rhs = [offset + inds[2] - 1, offset + inds[3]];
            rules.push((lhs, rhs));

            let remove_start = inds[0];
            let remove_end = inds[3] + 1;
            parent.drain(remove_start..remove_end);
            offset += remove_end - remove_start;
        }
    }
    rules
}

struct Rule {
    pattern: Vec<u8>,
    replacement: Vec<u8>,
    is_identity: bool,
}

fn extract_rules(code: &[u8]) -> Vec<Rule> {
    let parent = build_parent_list(code);
    greedy_pairs(parent)
        .into_iter()
        .map(|(m, r)| {
            let pattern = code[m[0]..m[1]].to_vec();
            let replacement = code[r[0]..r[1]].to_vec();
            let is_identity = pattern == replacement;
            Rule { pattern, replacement, is_identity }
        })
        .collect()
}

fn interpret_rules(rules: &[Rule], input: &[u8], max_step: usize) -> Vec<u8> {
    // Build Finders once per interpret call; they borrow `rules` which outlives the loop.
    let finders: Vec<memmem::Finder<'_>> =
        rules.iter().map(|r| memmem::Finder::new(&r.pattern)).collect();

    let mut tape = input.to_vec();

    'steps: for _ in 0..max_step {
        for (rule, finder) in rules.iter().zip(&finders) {
            if let Some(idx) = finder.find(&tape) {
                if rule.is_identity {
                    break 'steps;
                }
                tape.splice(
                    idx..idx + rule.pattern.len(),
                    rule.replacement.iter().copied(),
                );
                continue 'steps;
            }
        }
        break;
    }
    tape
}

#[pyfunction]
fn treemo(code: Vec<u8>, inp: Vec<u8>, max_step: usize) -> Vec<u8> {
    let rules = extract_rules(&code);
    interpret_rules(&rules, &inp, max_step)
}

#[pymodule]
fn treemo_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(treemo, m)?)?;
    Ok(())
}
