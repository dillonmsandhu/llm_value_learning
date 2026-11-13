"""Functional JAX program synthesis with reusable integer-polynomial registers."""
from typing import NamedTuple
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np
from .program_tasks import make_tasks, OPS


class ProgramState(NamedTuple):
    prompt_idx: jax.Array
    registers: jax.Array
    register_count: jax.Array
    step_count: jax.Array
    history_tokens: jax.Array
    done: jax.Array


class ProgramSynthesisEnv:
    VERSION='integer_polynomial_program_v1'
    def __init__(self,num_problems=20,max_steps=24,seed=42,eval_seed=12345,
                 program_min_ops=2,program_max_ops=4,program_degree=3,
                 program_max_inputs=6,program_registers=10,program_coefficient_limit=1024,
                 program_teacher_probability=.75,program_problem_file=None,
                 split='train',gamma=1.,**legacy):
        # Generic DAG launchers supply these options. They do not define this
        # environment's semantics or its derived action-space size.
        unexpected=set(legacy)-{'num_actions','num_milestones','variable_depth','difficulties'}
        if unexpected:raise TypeError(f'Unknown program environment options: {sorted(unexpected)}')
        for k,v,minimum in (('num_problems',num_problems,1),('max_steps',max_steps,2),
                            ('program_min_ops',program_min_ops,1),('program_max_ops',program_max_ops,1),
                            ('program_degree',program_degree,0),('program_max_inputs',program_max_inputs,2),('program_registers',program_registers,3),
                            ('program_coefficient_limit',program_coefficient_limit,5)):
            if type(v) is not int or v<minimum:raise ValueError(f'{k} must be an integer >= {minimum}')
        if program_min_ops>program_max_ops or program_max_ops+1>max_steps:raise ValueError('operation bounds must fit the horizon plus submission')
        if program_degree>6 or program_coefficient_limit>4096:raise ValueError('degree <=6 and coefficient limit <=4096 keep int32 arithmetic safe')
        if not 0<=program_teacher_probability<=1 or not 0<=gamma<=1:raise ValueError('probabilities and gamma must be in [0,1]')
        if split not in ('train','test'):raise ValueError('split must be train or test')
        self.num_problems,self.max_steps,self.gamma=num_problems,max_steps,float(gamma)
        self.eval_seed=eval_seed;self.degree=program_degree;self.limit=program_coefficient_limit
        self.max_inputs=program_max_inputs;self.max_ops=program_max_ops
        if program_registers<program_max_inputs+program_max_ops:
            raise ValueError('program_registers must cover program_max_inputs + program_max_ops')
        self.capacity=program_registers
        self.teacher_probability=float(program_teacher_probability)
        self.tasks=make_tasks(num_problems,seed,program_min_ops,program_max_ops,self.degree,
                              self.limit,self.max_inputs,split,program_problem_file)
        self.problem_max_steps=jnp.full(num_problems,max_steps,jnp.int32)
        self.thetas=jnp.array([t['reference_ops']/program_max_ops for t in self.tasks])
        C=self.degree+1;R=self.capacity
        inputs=np.zeros((num_problems,R,C),np.int32)
        plans=np.zeros((num_problems,self.max_ops,3),np.int32)
        plan_values=np.zeros_like(inputs)
        for p,t in enumerate(self.tasks):
            inputs[p,:len(t['inputs'])]=t['inputs']
            plans[p,:len(t['plan'])]=t['plan']
            plan_values[p,:len(t['program_values'])]=t['program_values']
        self.inputs=jnp.asarray(inputs);self.plans=jnp.asarray(plans);self.plan_values=jnp.asarray(plan_values)
        self.input_counts=jnp.array([len(t['inputs']) for t in self.tasks],jnp.int32)
        self.plan_lengths=jnp.array([len(t['plan']) for t in self.tasks],jnp.int32)
        self.targets=jnp.array([t['target'] for t in self.tasks],jnp.int32)
        self.binary_actions=4*R*R;self.submit_offset=self.binary_actions
        self.undo_action=self.submit_offset+R;self.reset_action=self.undo_action+1
        self.eos_action=self.submit_offset  # There are R explicit submit actions.
        self.num_actions=self.reset_action+1
        # Fixed prompt includes expression and initial register coefficients.
        # No target, reference plan, problem ID or correctness flag is encoded.
        self.PAD_TOKEN=0;self.BOS_TOKEN=1;self.SEP_TOKEN=2
        self.symbols={s:3+i for i,s in enumerate(('x','(',')','+','-','*','/','INPUTS','INVALID','UNDO','RESET'))}
        self.value_offset=32;self.action_offset=self.value_offset+2*self.limit+1
        self.vocab_size=self.action_offset+self.num_actions
        self.record_width=2+C  # Action, delimiter/status, computed coefficients.
        prefixes=[]
        for t in self.tasks:
            tokens=[self.BOS_TOKEN]
            for token in t['tokens']:
                tokens.append(self.value_offset+int(token)+self.limit if token.isdigit() else self.symbols[token])
            tokens += [self.SEP_TOKEN,self.symbols['INPUTS']]
            for row in t['inputs']:
                tokens += [self.value_offset+int(x)+self.limit for x in row]+[self.SEP_TOKEN]
            # Padded initial registers keep every problem's history start aligned.
            tokens += [self.SEP_TOKEN]*(self.max_inputs-len(t['inputs']))*(C+1)
            prefixes.append(tokens)
        self.prompt_len=8*self.max_ops+16+self.max_inputs*(C+1)
        if any(len(tokens)>self.prompt_len for tokens in prefixes):
            raise ValueError('expression tokenization exceeds fixed prompt bound')
        self.max_seq_len=self.prompt_len+max_steps*self.record_width
        prompt_array=np.zeros((num_problems,self.max_seq_len),np.int32)
        for p,tokens in enumerate(prefixes):
            # Interior padding uses SEP, so last-valid-token pooling remains valid.
            prompt_array[p,:self.prompt_len]=tokens+[self.SEP_TOKEN]*(self.prompt_len-len(tokens))
        self.prompts=jnp.asarray(prompt_array)
        self.config=dict(version=self.VERSION,num_problems=num_problems,max_steps=max_steps,
                         seed=seed,eval_seed=eval_seed,split=split,gamma=gamma,
                         program_min_ops=program_min_ops,program_max_ops=program_max_ops,
                         program_degree=program_degree,program_max_inputs=program_max_inputs,program_registers=program_registers,
                         program_coefficient_limit=program_coefficient_limit,
                         program_teacher_probability=program_teacher_probability,
                         program_problem_file=program_problem_file,capacity=R,num_actions=self.num_actions,
                         operator_names=list(OPS),reward='terminal exact answer only',
                         dataset_source='external' if program_problem_file else 'synthetic expression grammar',
                         fingerprints=[t['fingerprint'] for t in self.tasks])
        self.config['task_bank_hash']=hashlib.sha256(json.dumps(self.tasks,sort_keys=True).encode()).hexdigest()

    def reset_problem(self,index):
        index=jnp.asarray(index,jnp.int32)
        state=ProgramState(index,self.inputs[index],self.input_counts[index],jnp.array(0,jnp.int32),
                           self.prompts[index],jnp.array(False))
        return state.history_tokens,state

    def reset(self,rng,params=None):
        return self.reset_problem(jax.random.randint(rng,(),0,self.num_problems))

    def encode_binary(self,op,i,j):return op*self.capacity*self.capacity+i*self.capacity+j

    def step(self,rng,state,action,params=None):
        del rng,params
        R=self.capacity;C=self.degree+1
        action=jnp.asarray(action,jnp.int32)
        op=jnp.clip(action//(R*R),0,3);i=jnp.clip((action//R)%R,0,R-1);j=jnp.clip(action%R,0,R-1)
        a,b=state.registers[i],state.registers[j]
        # jnp.convolve promotes integer inputs to float: use exact int32
        # multiply/add so mathematical equality and branch targets stay exact.
        powers=jnp.arange(C)[:,None]+jnp.arange(C)[None,:]
        product=jnp.zeros(2*C-1,jnp.int32).at[powers].add(a[:,None]*b[None,:])
        denominator=jnp.where(b[0]!=0,b[0],1)
        quotient=a//denominator
        candidates=jnp.stack([a+b,a-b,product[:C],quotient])
        value=candidates[op]
        domain_ok=jnp.where(op==2,jnp.all(product[C:]==0),True)
        domain_ok &= jnp.where(op==3,(b[0]!=0)&jnp.all(b[1:]==0)&jnp.all(a%denominator==0),True)
        binary=(action>=0)&(action<self.binary_actions)
        valid_binary=binary&(i<state.register_count)&(j<state.register_count)&(state.register_count<R)&domain_ok&jnp.all(jnp.abs(value)<=self.limit)
        slot=jnp.minimum(state.register_count,R-1)
        registers=jnp.where(valid_binary,state.registers.at[slot].set(value),state.registers)
        count=state.register_count+valid_binary.astype(jnp.int32)
        undo=(action==self.undo_action)&(state.register_count>self.input_counts[state.prompt_idx])
        registers=jnp.where(undo,registers.at[jnp.maximum(count-1,0)].set(jnp.zeros(C,jnp.int32)),registers)
        count-=undo.astype(jnp.int32)
        reset=action==self.reset_action
        registers=jnp.where(reset,self.inputs[state.prompt_idx],registers)
        count=jnp.where(reset,self.input_counts[state.prompt_idx],count)
        submit=(action>=self.submit_offset)&(action<self.submit_offset+R)
        selected=jnp.clip(action-self.submit_offset,0,R-1)
        correct=submit&(selected<state.register_count)&jnp.all(state.registers[selected]==self.targets[state.prompt_idx])
        done=submit|(state.step_count+1>=self.max_steps)
        valid=valid_binary|undo|reset|(submit&(selected<state.register_count))
        # Explicit public operation result; invalid/domain failures have no
        # hidden semantic feedback. Goal correctness appears only at terminal.
        status=jnp.where(valid,self.SEP_TOKEN,self.symbols['INVALID'])
        encoded_value=jnp.clip(value,-self.limit,self.limit)+self.value_offset+self.limit
        record=jnp.concatenate([jnp.array([self.action_offset+jnp.clip(action,0,self.num_actions-1),status]),
                                 jnp.where(valid_binary,encoded_value,jnp.full(C,self.SEP_TOKEN))])
        start=self.prompt_len+jnp.minimum(state.step_count,self.max_steps-1)*self.record_width
        history=jax.lax.dynamic_update_slice(state.history_tokens,record,(start,))
        next_state=ProgramState(state.prompt_idx,registers,count,state.step_count+1,history,done)
        next_state=jax.tree_util.tree_map(lambda new,old:jnp.where(state.done,old,new),next_state,state)
        reward=(~state.done&correct).astype(jnp.float32)
        return next_state.history_tokens,next_state,reward,next_state.done,dict(invalid_action=(~state.done)&~valid)

    def reference_action(self,state):
        # A solution demonstrator, not an observation feature or critic oracle.
        p=state.prompt_idx;R=self.capacity
        available=jnp.arange(R)<state.register_count
        goal=available&jnp.all(state.registers==self.targets[p],axis=-1)
        def match(value):
            matches=available&jnp.all(state.registers==value,axis=-1)
            return jnp.any(matches),jnp.argmax(matches)
        def candidate(k):
            op,i,j=self.plans[p,k]
            have_i,ii=match(self.plan_values[p,i]);have_j,jj=match(self.plan_values[p,j])
            have_result,_=match(self.plan_values[p,self.input_counts[p]+k])
            ready=(k<self.plan_lengths[p])&have_i&have_j&~have_result
            return ready,self.encode_binary(op,ii,jj)
        ready,actions=jax.vmap(candidate)(jnp.arange(self.max_ops))
        next_action=jnp.where(jnp.any(ready)&(state.register_count<R),actions[jnp.argmax(ready)],self.reset_action)
        return jnp.where(jnp.any(goal),self.submit_offset+jnp.argmax(goal),next_action).astype(jnp.int32)

    def action_probabilities(self,state):
        R=self.capacity
        ids=jnp.arange(self.num_actions);binary=ids<self.binary_actions
        i=(ids//R)%R;j=ids%R
        # Exploration support depends on public register availability, not
        # whether an action advances the hidden reference solution.
        valid=binary&(i<state.register_count)&(j<state.register_count)&(state.register_count<R)
        valid |= (ids>=self.submit_offset)&(ids<self.submit_offset+state.register_count)
        valid |= (ids==self.undo_action)&(state.register_count>self.input_counts[state.prompt_idx])
        valid |= ids==self.reset_action
        uniform=valid/jnp.maximum(valid.sum(),1)
        teacher=jax.nn.one_hot(self.reference_action(state),self.num_actions)
        probs=self.teacher_probability*teacher+(1-self.teacher_probability)*uniform
        return jnp.where(state.done,jax.nn.one_hot(self.submit_offset,self.num_actions),probs)

    def sample_base_action(self,rng,state,params=None):
        return jax.random.categorical(rng,jnp.log(self.action_probabilities(state))).astype(jnp.int32)

    def export_tasks(self,path):
        from pathlib import Path
        Path(path).write_text(''.join(json.dumps(t,sort_keys=True)+'\n' for t in self.tasks))
