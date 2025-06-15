from config.arguments import args      

class BaseModel:
    def __init__(self, 
                 env,
                 thousands=1e-5,
                 gamma=0.9,
                 model_path = None,
                 env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):
        self.env = env
        self.thousands = thousands
        self.gamma = gamma
        
        self.model_path = model_path
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step
