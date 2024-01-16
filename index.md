# <p align='center'>Short Answers</p>
# <p align='center'> Unit - 1</p>

1. **Define: AI**
   - AI, or *Artificial Intelligence*, refers to the development of computer systems that can perform tasks that typically require human intelligence. These tasks include problem-solving, learning, understanding natural language, recognizing patterns, and adapting to new situations.

2. **Applications of AI:**
   - AI has various applications, including:
     - *Natural Language Processing (NLP):* Language translation, chatbots.
     - *Computer Vision:* Image and video analysis, facial recognition.
     - *Machine Learning:* Predictive analytics, recommendation systems.
     - *Robotics:* Autonomous vehicles, industrial automation.
     - *Healthcare:* Diagnosis, personalized medicine.
     - *Gaming:* Game playing, procedural content generation.

3. **Role of an Agent:**
   - An agent is an entity that perceives its environment through sensors and acts upon that environment through actuators. The role of an agent is to make decisions and take actions in its environment to achieve its goals.

4. **Agent Program:**
   - The agent program is the implementation of an agent's design. It defines the mapping from percept sequences to actions. It is the software or algorithm that guides the agent's behavior based on its observations of the environment.

5. **Types or Properties of Task Environment:**
   - **Fully Observable vs. Partially Observable:** Whether the agent can see the complete state of the environment at any given time.
   - **Deterministic vs. Stochastic:** Whether the next state is completely determined by the current state and action.
   - **Episodic vs. Sequential:** Whether the agent's experience is divided into episodes or is a continuous process.
   - **Static vs. Dynamic:** Whether the environment changes while the agent is deliberating.

6. **Intelligent Agent:**
   - An intelligent agent is an entity that perceives its environment, reasons about it, and acts upon it to achieve goals. It can exhibit behaviors that, when performed by a human, would be considered intelligent.

7. **Problem Formulation, Well-Defined Problem:**
   - **Problem Formulation:** It involves specifying the problem, initial state, actions, and goal state. It is the first step in solving problems using AI techniques.
   - **Well-Defined Problem:** A problem is well-defined when the initial state, actions, and goal state are precisely specified, allowing for a clear and unambiguous solution.

8. **Different Types of Tree Traversals:**
   - **Inorder:** Visit left subtree, visit root, visit right subtree.
   - **Preorder:** Visit root, visit left subtree, visit right subtree.
   - **Postorder:** Visit left subtree, visit right subtree, visit root.

9. **State Space Representation:**
   - State space representation is a way to model the possible states of a system and the transitions between them. It's often used in AI problem-solving to represent the different configurations a problem can have and the possible actions to move between them.

10. **Different Types of Problems:**
    - **Well-Defined Problems:** Problems with precisely specified initial states, actions, and goal states.
    - **Ill-Defined Problems:** Problems with unclear or ambiguous specifications.
    - **Single-Agent Problems:** Problems where only one agent is involved.
    - **Multi-Agent Problems:** Problems involving multiple agents, each with its goals and actions.


# <p align='center'> Unit - 2</p>

1. **Components of Search Problem:**
   - The components of a search problem typically include:
     - **Initial State:** The starting configuration of the problem.
     - **Actions:** The set of possible moves or operations.
     - **Transition Model:** Describes the result of each action.
     - **Goal Test:** Determines if a state is a goal state.
     - **Path Cost:** Assigns a cost to each path.

2. **Define: Random Search:**
   - Random Search is an uninformed search algorithm where the system explores the search space without any specific order or strategy. It selects actions randomly until it finds a solution or a stopping criterion is met.

3. **Informed and Uninformed Search Techniques:**
   - **Informed:**
     - *Pure Hieuristic*
     - *A* (A-star) Search
     - *Best First Search*
   - **Uninformed:**
     - *BFS* (Breadth-First Search)
     - *DFS* (Depth-First Search)

4. **Differences between BFS and DFS:**
   - **BFS (Breadth-First Search):**
     - Explores neighbor nodes before moving to the next level.
     - Uses a queue data structure.
   - **DFS (Depth-First Search):**
     - Explores as far as possible along one branch before backtracking.
     - Uses a stack or recursion.

5. **Define: Search Heuristic:**
   - A search heuristic is an informed search strategy that provides an estimate of the cost or distance from a given state to the goal state. It guides the search algorithm to prioritize paths that seem more promising based on the heuristic evaluation.

6. **Heuristic Function:**
   - A heuristic function is a function that estimates the cost or value of reaching the goal from a given state in a search problem. It guides the search algorithm by providing a heuristic evaluation of different states.

7. **Game Playing:**
   - Game playing refers to the development of intelligent agents that can make decisions in a competitive environment, typically games. Characteristics include:
     - **Adversarial Nature:** Involves competition against an opponent.
     - **Sequential Decisions:** Players take turns making moves.
     - **Uncertainty:** Limited information about the opponent's moves.

8. **Search Tree for Tic-Tac-Toe:**
   - Unfortunately, I can't draw here, but a search tree for Tic-Tac-Toe would represent all possible moves and their consequences, branching out based on player actions and reactions.

9. **Fitness Number:**
   - The fitness number represents the quality or suitability of an individual in an evolutionary algorithm. It is a measure of how well an individual solution meets the objectives or criteria set by the optimization problem.

# <p align='center'> Unit - 3</p>

1. **Define: Probabilistic Reasoning, Uncertainty:**
   - **Probabilistic Reasoning:** Probabilistic reasoning is a form of reasoning that deals with uncertainty. It involves using probability theory to represent and manipulate uncertain information.
   - **Uncertainty:** Uncertainty refers to the lack of complete knowledge or certainty about the outcome of events. In probabilistic reasoning, uncertainty is often represented using probability distributions.

2. **Define: Probability. Mention its types:**
   - **Probability:** Probability is a measure of the likelihood of an event occurring. It ranges from 0 (impossible) to 1 (certain). Types of probability include:
     - *Marginal Probability:* Probability of an event irrespective of the occurrence of other events.
     - *Conditional Probability:* Probability of an event given that another event has occurred.
     - *Joint Probability:* Probability of the intersection of two events.

3. **Axioms or Rules of Probability:**
   - The axioms of probability include:
     - *Non-Negativity:* \(P(A) \geq 0\) for any event \(A\).
     - *Normalization:* \(P(\text{Sample Space}) = 1\).
     - *Additivity:* \(P(A \cup B) = P(A) + P(B)\) if \(A\) and \(B\) are mutually exclusive.

4. **Define: Conditional Probability:**
   - Conditional Probability is the probability of an event occurring given that another event has already occurred. Mathematically, it is denoted as \(P(A|B)\), read as "the probability of A given B."

5. **What is Probability Model?**
   - A Probability Model is a mathematical representation of a real-world process involving uncertainty. It consists of a sample space, events, and probability assignments to events, providing a framework to describe and analyze uncertain situations.

6. **Differentiate Independent and Dependent Event:**
   - **Independent Events:** The occurrence of one event does not affect the occurrence of the other.
   - **Dependent Events:** The occurrence of one event affects the occurrence of the other.

7. **Define: Bayes Rule:**
   - Bayes' Rule is a formula that relates the conditional and marginal probabilities of random events. It is expressed as \(P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}\) and is fundamental in Bayesian inference.

8. **Why do we need temporal probabilistic model?**
   - Temporal probabilistic models are needed to model systems where events unfold over time, and the probability of an event may depend on the history of events leading up to it. They are essential in representing dynamic and evolving systems.

9. **What is Bayesian (Belief) Network? List out the applications of its:**
   - A Bayesian Network is a graphical model that represents probabilistic relationships among a set of variables. Applications include:
     - Medical Diagnosis
     - Risk Assessment
     - Speech Recognition
     - Fraud Detection

10. **Types of Probabilistic Models:**
    - **Bayesian Networks:** Graphical models representing probabilistic relationships.
    - **Markov Models:** Models where the future state depends only on the current state.
    - **Hidden Markov Models:** Extend Markov models to account for hidden states.

11. **Define: Markov Chain:**
    - A Markov Chain is a mathematical model that represents a system whose state transitions follow the Markov property, where the probability of transitioning to any particular state depends solely on the current state and time elapsed, not on the sequence of events that preceded it.


# <p align='center'> Unit - 4 </p>


1. **What is MDP?**
   - **MDP (Markov Decision Process):** MDP is a mathematical framework used for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. It's widely used in reinforcement learning.

2. **Components of MDP:**
   - - **State Space:** Set of all possible states.
     - **Action Space:** Set of all possible actions.
     - **Transition Probability Function:** Defines the probability of transitioning from one state to another given an action.
     - **Reward Function:** Specifies the immediate reward associated with a state-action pair.
     - **Policy:** A strategy that defines the decision-making rules.

3. **What is Markov Property?**
   - **Markov Property:** The future state of a system (or process) depends only on the present state and not on the sequence of events that preceded it. It is the memoryless property that characterizes Markov processes.

4. **Define: Utility Function:**
   - **Utility Function:** A utility function assigns a numerical value to each possible outcome, expressing the decision-maker's preferences. It helps in choosing the most favorable outcomes among various possibilities.

5. **Properties of Markov Process:**
   - **Memoryless Property:** Future states depend only on the current state.
   - **Markovian Transition Probability:** Probability distribution for state transitions is determined by the current state and action.

6. **What is Optimal Policy?**
   - **Optimal Policy:** The policy that maximizes the expected cumulative reward over time in a Markov Decision Process. It is the best strategy for an agent to follow.

7. **Define: Policy Iteration:**
   - **Policy Iteration:** An iterative algorithm used to find the optimal policy in a Markov Decision Process. It alternates between policy evaluation (determining the value of states under the current policy) and policy improvement (changing the policy to be more greedy).

8. **What is Value Iteration?**
   - **Value Iteration:** An algorithm for finding the optimal value function and policy in a Markov Decision Process. It iteratively updates the value function until it converges to the optimal values.

9. **Difference between Markov Model and Hidden Markov Model:**
   - **Markov Model:** Represents a system where the state is directly observable.
   - **Hidden Markov Model (HMM):** Represents a system where the state is not directly observable but generates observable outcomes.

10. **Types of Axioms in Utility Theory:**
   - - **Completeness Axiom:** Assumes that the decision-maker can compare and rank all possible outcomes.
      - **Transitivity Axiom:** Assumes that if an outcome is preferred to a second, and the second is preferred to a third, then the first must be preferred to the third.
      - **Independence Axiom:** Assumes that the preference between two outcomes is not affected by the introduction of a third, irrelevant option.

11. **Components of POMDP:**
   - **State Space:** Set of all possible states.
     - **Observation Space:** Set of all possible observations.
     - **Action Space:** Set of all possible actions.
     - **Transition Probability Function:** Defines the probability of transitioning from one state to another given an action.
     - **Observation Probability Function:** Specifies the probability of observing a particular outcome given the current state and action.
     - **Reward Function:** Specifies the immediate reward associated with a state-action pair.
     - **Discount Factor:** Represents the importance of future rewards in the decision-making process.

# <p align='center'> Unit - 5</p>

1. **What is Reinforcement Learning?**
   - **Reinforcement Learning:** Reinforcement Learning is a type of machine learning paradigm where an agent learns how to behave in an environment by performing actions and receiving rewards. The agent learns to make decisions by interacting with the environment to maximize cumulative rewards.

2. **Key Features of Reinforcement Learning:**
   - - **Trial-and-Error Learning:** The agent learns by trying different actions and observing their consequences.
     - **Delayed Rewards:** The consequences of actions may not be immediately apparent; rewards may be delayed.
     - **Exploration and Exploitation:** Balancing the exploration of new actions and exploiting known actions is crucial.
     - **Sequential Decision Making:** Decisions are made over a sequence of actions and states.

3. **Approaches Used to Implement Reinforcement Learning:**
   - - **Value-Based Methods:** Learn a value function that estimates the expected cumulative reward of being in a state or taking an action.
     - **Policy-Based Methods:** Directly learn a policy that maps states to actions.
     - **Model-Based Methods:** Learn a model of the environment and use it for decision-making.

4. **Elements of Reinforcement Learning:**
   - - **Agent:** Learns from the environment and takes actions.
     - **Environment:** The external system with which the agent interacts.
     - **State:** A representation of the current situation.
     - **Action:** The decision or move made by the agent.
     - **Reward:** Feedback from the environment indicating the desirability of the action.

5. **Differentiate Passive Reinforcement Learning and Active Reinforcement Learning:**
   - - **Passive Reinforcement Learning:** The agent observes and learns from a fixed policy without the ability to influence or select actions.
     - **Active Reinforcement Learning:** The agent actively explores the environment and learns a policy by selecting actions and receiving feedback.

6. **Applications of Reinforcement Learning:**
   - - **Game Playing:** AlphaGo, chess, etc.
     - **Robotics:** Control and decision-making for robotic systems.
     - **Autonomous Vehicles:** Navigation and decision-making for self-driving cars.
     - **Finance:** Portfolio optimization and trading strategies.

7. **Differentiate Reinforcement Learning and Supervised Learning:**
   - - **Reinforcement Learning:** Learns from interaction with the environment, receiving feedback in the form of rewards or penalties. No explicit guidance is provided on the correct actions.
     - **Supervised Learning:** Learns from a labeled dataset where each input is associated with a corresponding output. The model is trained to map inputs to predefined outputs.

8. **Common Active and Passive RL Techniques:**
   - - **Active RL Techniques:** Q-learning, Deep Q Network (DQN), Policy Gradients.
     - **Passive RL Techniques:** Monte Carlo methods, Temporal Difference learning.

9. **Define: Q-Learning:**
   - **Q-Learning:** Q-learning is a model-free reinforcement learning algorithm that aims to learn a quality function (Q-function) representing the expected cumulative reward for taking a particular action in a given state.

10. **What is Deep Q Neural Network (DQN)?**
   - **Deep Q Neural Network (DQN):** DQN is an extension of Q-learning that uses a deep neural network to approximate the Q-function. It is employed to handle high-dimensional state spaces, enabling reinforcement learning in complex environments.
