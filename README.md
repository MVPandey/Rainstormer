# Rainstormer

A self-improving AI agent repo with the end goal of incorporating Monte Carlo Tree Search for context-engineering and iterative refinement to help explore ideas/research/brainstorm through simulation and backpropagation.

## Why MCTS?

Monte Carlo Tree Search gives us a principled way to balance exploration and exploitation when searching through possible context configurations. It's been incredibly successful in game playing and planning domains, and I think the same properties that make it work there—handling large search spaces, learning from rollouts, building up knowledge incrementally—should apply to context optimization, etc. 

The hypothesis is that we can treat context engineering as a search problem where each node represents a different context configuration, and rollouts give us signal about which configurations/set of contexts can lead to better agent performance. 

## Current status

Building the basic MCTS implementation and figuring out how to represent context configurations as search tree nodes. Most of what's here right now is exploratory and will probably change as I figure out what actually works.

Goal eventually is something like -> create chatnbot -> save context over time/learn what the user considers ideal/non-ideal -> do heavy reasoning with MCTS -> save context -> distill learnings into a dataset for dynamic loras (potentially)

For now, we're focusing on one-off idea brainstorming

## What I'm not doing (yet)

I'm explicitly avoiding external dependencies on existing agentic frameworks for now. Most of them make assumptions about architecture or add complexity that I don't need yet. Once I have the core working and understand the problem better, I might integrate with other tools, but starting simple lets me actually understand what's happening.