---
layout: default
title:  Proposal
---
## Summary of the Project

The classic Snake Game traditionally consists of one snake collecting as many apples as it can without running into a wall or itself. The action space is {up, down, left, and right}, with one of those actions being impossible to do based on the direction of the snake, as you can't turn back into yourself 180 degrees. In our project, we decided to implement two to three snake agents in the grid environment instead of just one snake agent, each running its own algorithm out of this list: Policy Optimization (PPO), Group Sequence Policy Optimization (GSPO), and Group Relative Policy Optimization (GRPO). The snake agents are playing against each other in our project, where they are trying to maximize their score and minimize their opponents' scores, which is what our project analysis will mainly focus on.

## Project goals

* **Minimum Goal:** Implementing a two-agent Snake game where the agents are against each other would be the minimum amount to get finished in this project. For this goal, we expect for both snake agents to reach a length of at least 1/8th of the area of the grid each while still staying alive.

* **Realistic goal:** Implementing a three-agent Snake game where the agents are against each other would be a realistic amount of work we could finish for this project. For this goal, we expect each snake agent to reach a length of at least 1/6th of the area of the grid while still staying alive. We also expect to set rewards in such a way that the snake agents try to block each other off intentionally in order to destroy another snake agent or stop them from getting more points through eating apples.

* **Moonshot goal:** Modifying the grid so that snakes are able to go through walls and teleport to the wall opposite of the wall they went through would be a moonshot modification we will keep in mind if time allows, as this modification of the grid will greatly change the strategies the snake agents would have to use in order to stay alive and try to sabotage the other snake agents.

## AI/ML Algorithms

As the action space is discrete, the environment is stochastic, the horizon is episodic, and hyperparameters don't need to be constantly hypertuned, Proximal Policy Optimization (PPO) is the main algorithm we will be using on one of the snake agents, with Group Sequence Policy Optimization (GSPO) and Group Relative Policy Optimization (GRPO) also being similar algorithms we could use for the other snake agents. The minimax algorithm could also be potentially used to implement the opponent aspect between the snake agents.

## Evaluation Plan

Initially, we will implement the Proximal Policy Optimization (PPO), Policy Optimization (GSPO), and Group Relative Policy Optimization (GRPO) algorithms all separately in a single-agent system, where the goal is for each snake to collect one fourth of the apples on the grid (ex.: 25 apples on a 10x10 grid). We will then have our team members play five rounds of the Snake game each and get the average score between all fifteen total rounds. Once we get to a point where all agents beat our average score by around 25-35% we will implement a Minimax algorithm in order to have the agents go against each other in a multi-agent system.

In terms of toy cases, we plan to scale the grid and analyze how the behavior of each agent cahnges, scaling the grid from an initial start of a 5x5 grid all the way up to a 25x25 grid. We plan to display the board in a virtual UI screen and constantly show the real-time states of each snake agent and the fruits to collect. We also plan to show the current score of each agent currently on the board in order to compare the scores, and even potentially show the time at which a snake agent ran into a wall or got destroyed by another snake agent. We will know that we successfully implemented the algorithms and tuned their hyperparameters correctly if they consistently go to the apples on the grid in an efficient path while also actively rerouting their path to avoid the other snake agents or even to cut the other agents off to try to trap/destroy them.

## AI Tool Usage

ChatGPT was used for potential gameplay modification aspects to add to the Snake game, such as being able to teleport through walls. Potential algorithm ideas were also provided by ChatGPT, in which we researched some of these algorithm ideas through articles in order to decide if these algorithms were viable or not.