from spider_agent.agent.agents import PromptAgent

class ReFoRCEAgent(PromptAgent):
    """Implementation of the ReFoRCE (Self-Refinement Agent with Format 
    Restriction and Column Exploration) approach."""
    
    def __init__(
        self,
        model="qwq_self",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        parallel_count=3,  # Number of parallel threads
        max_refinement_iterations=5,  # Max iterations for self-refinement
        use_plan=False
    ):
        super().__init__(model, max_tokens, top_p, temperature, 
                          max_memory_length, max_steps, use_plan)
        self.parallel_count = parallel_count
        self.max_refinement_iterations = max_refinement_iterations