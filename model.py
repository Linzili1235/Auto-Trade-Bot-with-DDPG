import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# WHETHER USE GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ACTOR NETWORK: CONTINUOUS ACTION
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return F.tanh(self.l3(a)) # USE TANH AS ACTIVATION FUNCTION TO MAP THE DATE INTO [-1, 1]


# CRITIC NETWORK: CRITISIZE THE VALUE OF AN ACTION
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.concat([state, action], 1)))
        q = F.relu(self.l2(q))
        return self.l3(q)


# DDPG MODEL   
class DDPGModel(object):
    def __init__(self, state_dim, action_dim, max_action, gamma = 0.99, tau = 0.005):
        # ACTOR AND TARGET ACTOR NETWORK
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=1e-4)

        # CRITIC AND TARGET CRITIC NETWORK
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), weight_decay=1e-2)

        self.gamma = gamma
        self.tau = tau


    # SELECT AN ACTION BASED ON THE CURRENT STATE
    def select_action(self, state):
        
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=device)
            return self.actor(state).numpy().flatten()


    # TRAIN THE NETWORK
    def train(self, replay_buffer, batch=64):
        # SAMPLE FROM THE REPLAY BUFFER
        state, action, next_state, reward, done = replay_buffer.sample(batch)

        # COMPUTE THE Q VALUE FROM TARGET NETWORK
        q_target = self.critic_target(next_state, self.actor_target(next_state).detach())
        q_target = reward + ((1- done) * self.gamma * q_target)

        # COMPUTE THE Q VALUE FROM CRITIC NETWORK
        q_eval = self.critic(state, action)

        # COMPUTE THE LOSS FROM CRITIC NETWORK
        critic_loss = nn.MSELoss()(q_eval, q_target)
        # print(critic_loss)

        # UPDATE THE PARAMETERS
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # COMPUTE THE LOSS FROM CRITIC NETWORK
        actor_loss = -self.critic(state, self.actor(state)).mean()
        # print(actor_loss)

        # UPDATE THE PARAMETERS
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
           
        # GET THE CURRENT WEIGHTS OF EACH NETWORK
        critic_target_state_dict = self.critic_target.state_dict()
        critic_state_dict = self.critic.state_dict()
        
        # CALCULATE SOFT UPDATE OF TARGET NET WEIGHTS
        # θ′ ← τ θ + (1 −τ )θ′
        for key in critic_state_dict:
            critic_target_state_dict[key] = critic_state_dict[key]*self.tau + critic_target_state_dict[key]*(1-self.tau)
        
        # APPLY THE UPDATE TO THE TARGET NETWORK
        self.critic_target.load_state_dict(critic_target_state_dict)


        # GET THE CURRENT WEIGHTS OF EACH NETWORK
        actor_target_state_dict = self.actor_target.state_dict()
        actor_state_dict = self.actor.state_dict()
        
        # CALCULATE SOFT UPDATE OF TARGET NET WEIGHTS
        # θ′ ← τ θ + (1 −τ )θ′
        for key in actor_state_dict:
            actor_target_state_dict[key] = actor_state_dict[key]*self.tau + actor_target_state_dict[key]*(1-self.tau)
        
        # APPLY THE UPDATE TO THE TARGET NETWORK
        self.actor_target.load_state_dict(actor_target_state_dict)


        # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        #     target_param.data.copy_(target_param * (1.0 - self.tau) + param * self.tau)
        # for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        #     target_param.data.copy_(target_param * (1.0 - self.tau) + param * self.tau)




    # SAVE THE MODEL PARAMETERS   
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + '_critic')
        torch.save(self.critic_optimizer.state_dict(), filename + '_critic_optimizer')

        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.actor_optimizer.state_dict(), filename + '_actor_optimizer')
        

    # LOAD THE MODEL PARAMETERS
    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + '_critic'))
        self.critic_optimizer.load_state_dict(torch.load(filename + '_critic_optimizer'))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + '_actor'))
        self.actor_optimizer.load_state_dict(torch.load(filename + '_actor_optimizer'))
        self.actor_target = copy.deepcopy(self.actor)

        
