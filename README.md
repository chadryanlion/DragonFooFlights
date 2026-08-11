# DragonFooFlights
DragonFooFlight learns cooperative adaptive flight dynamics with DRL and RBV tidepool and Fractal Coastline Obstacle Optimization

Dragonfly DRL

A research-oriented reinforcement-learning environment for studying adaptive exploration, behavioral repertoires, and structured control under complex environmental dynamics.

Project hypothesis

This project investigates whether structured state representations and adaptive exploration mechanisms can produce qualitatively different behavioral repertoires than conventional epsilon-greedy reinforcement learning.

The initial environment represents environmental state through radial-basis features derived from a synthetic tide-pool/coastline system. These representations are evaluated with baseline DQN and NoisyNet DQN agents, followed by count-based exploration and adaptive exploration calibration.

Later experiments will compare multiple deep reinforcement-learning algorithms and evaluate their resulting behavioral repertoires using distributional divergence, hierarchical clustering, and control-performance metrics.

Initial research sequence
Establish a conventional DQN baseline.
Replace epsilon-greedy exploration with NoisyNet exploration.
Add state-visitation counts.
Introduce adaptive exploration-temperature/rheostat control.
Evaluate the exploration controller using conjugate-gradient optimization.
Introduce coastline/fractal environmental descriptors.
Compare learned flight-like behavioral repertoires.
Compare dragonfly-inspired swarm behavior, monarch-inspired migration, and a no-flight control condition.
Evaluate multiple DRL algorithms under equivalent environments and metrics.
Behavioral representation

The project treats an agent's behavior as a repertoire rather than a single scalar reward. Trajectories are represented by distributions over speed, heading, altitude, turning behavior, acceleration, neighborhood distance, obstacle distance, and energy expenditure.

Behavioral similarity is subsequently evaluated using divergence measures and hierarchical clustering.

Neuroscience-inspired control layer

A later experimental layer will investigate functional analogies between reinforcement-learning control mechanisms and mammalian motor/value architectures, including action gating, prediction-error correction, conflict monitoring, planning, value/context integration, and salience.

These are computational analogies rather than claims of one-to-one correspondence with anatomical structures.

Status

Early development.

The current priority is establishing reproducible baselines before integrating the larger archipelago, FalseOrb, flight-repertoire, and Mars-task components.
