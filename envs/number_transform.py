"""Digit-observed number planning with exact arithmetic and a fixed deadline."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

NUMBER_DEFAULTS=dict(number_min_value=0,number_max_value=999,number_min_distance=2,
                     number_max_distance=5,number_teacher_probability=.75,number_split_mode='pair',number_stop=1)


def add_number_arguments(parser):
    for name,default in NUMBER_DEFAULTS.items():
        kwargs=dict(type=type(default),default=default)
        if name=='number_stop':kwargs.update(choices=(0,1),help='Enable explicit STOP submission (default 1); 0 keeps the fixed-duration control')
        parser.add_argument('--'+name,**kwargs)


def number_kwargs(args):return {name:getattr(args,name) for name in NUMBER_DEFAULTS}


@lru_cache(maxsize=4)
def search_graph(minimum,maximum):
    """All-pairs shortest distances in the bounded arithmetic graph, NOT V^pi."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    values=np.arange(minimum,maximum+1,dtype=np.int32)
    x=values[:,None];digits=np.arange(1,10,dtype=np.int32)[None,:]
    candidates=np.concatenate([x+digits,x-digits,x*digits,x//digits],axis=1)
    valid=(candidates>=minimum)&(candidates<=maximum)
    valid[:,27:] &= x%digits==0
    following=np.where(valid,candidates,x).astype(np.int32)
    row,col=np.nonzero(valid&(following!=x))
    graph=csr_matrix((np.ones(len(row),np.float32),(row,following[row,col]-minimum)),shape=(len(values),len(values)))
    distance=shortest_path(graph,directed=True,unweighted=True,method='D')
    if not np.isfinite(distance).all():raise ValueError('bounded arithmetic graph must be connected')
    distance=distance.astype(np.uint16)
    # Cached host arrays are read-only; no environment can mutate another's graph.
    for a in (following,valid,distance):a.setflags(write=False)
    return following,valid,distance


def test_membership(starts,targets,mode):
    # A stable value-based hash: pair membership does not change with seed,
    # sampling, or the index assigned to a task in its selected distance bucket.
    a=np.asarray(starts,dtype=np.int64)+32768
    b=np.asarray(targets,dtype=np.int64)+32768
    if mode=='target':a=np.zeros_like(a)
    h=(a.astype(np.uint64)*73856093)^(b.astype(np.uint64)*19349663)
    h &= np.uint64(0xffffffff);h ^= h>>np.uint64(16)
    h=(h*np.uint64(2246822519))&np.uint64(0xffffffff);h ^= h>>np.uint64(13)
    return h%np.uint64(5)==0


class NumberState(NamedTuple):
    prompt_idx: jax.Array
    current_value: jax.Array
    target_value: jax.Array
    step_count: jax.Array
    history_tokens: jax.Array
    done: jax.Array


class NumberTransformEnv:
    VERSION='number_transform_submit_v2'
    def __init__(self,max_steps=8,seed=42,eval_seed=12345,split='train',gamma=1.,
                 number_min_value=0,number_max_value=999,number_min_distance=2,
                 number_max_distance=5,number_teacher_probability=.75,number_split_mode='pair',number_stop=1,**legacy):
        unexpected=set(legacy)-{'num_problems','num_actions','num_milestones','variable_depth','difficulties'}
        if unexpected:raise TypeError(f'Unknown number environment options: {sorted(unexpected)}')
        for name,v in (('max_steps',max_steps),('number_min_value',number_min_value),('number_max_value',number_max_value),
                       ('number_min_distance',number_min_distance),('number_max_distance',number_max_distance)):
            if type(v) is not int:raise ValueError(f'{name} must be an integer')
        if not -9999<=number_min_value<number_max_value<=9999 or number_max_value-number_min_value+1>4096:
            raise ValueError('require -9999 <= min < max <=9999 and at most 4096 integer states')
        if not 1<=number_min_distance<=number_max_distance<=max_steps:
            raise ValueError('require 1 <= min_distance <= max_distance <= max_steps')
        if max_steps>99:raise ValueError('max_steps must be <=99')
        if not 0<=number_teacher_probability<=1 or not 0<=gamma<=1:raise ValueError('probabilities and gamma must be in [0,1]')
        if split not in ('train','test') or number_split_mode not in ('pair','target'):raise ValueError('invalid split or split mode')
        if type(number_stop) not in (bool,int) or number_stop not in (0,1):raise ValueError('number_stop must be 0 or 1')
        self.stop_enabled=bool(number_stop)
        self.stop_action=36 if self.stop_enabled else None
        self.max_steps=max_steps;self.gamma=float(gamma);self.eval_seed=eval_seed
        self.minimum=number_min_value;self.maximum=number_max_value
        following,valid,distance=search_graph(self.minimum,self.maximum)
        self.next_values=jnp.asarray(following);self.valid_actions=jnp.asarray(valid);self.distances=jnp.asarray(distance)
        self.teacher_probability=float(number_teacher_probability)
        starts=[];targets=[];levels=[];counts=[];offsets=[]
        for d in range(number_min_distance,number_max_distance+1):
            s,t=np.nonzero(distance==d);s=s+self.minimum;t=t+self.minimum
            selected=test_membership(s,t,number_split_mode)==(split=='test')
            s,t=s[selected],t[selected]
            if not len(s):continue
            offsets.append(sum(counts));counts.append(len(s));levels.append(d)
            starts.append(s);targets.append(t)
        if not counts:raise ValueError('no pairs match the selected distance range and split')
        self.starts=jnp.asarray(np.concatenate(starts),jnp.int32)
        self.targets=jnp.asarray(np.concatenate(targets),jnp.int32)
        self.bucket_offsets=jnp.asarray(offsets,jnp.int32);self.bucket_counts=jnp.asarray(counts,jnp.int32)
        self.levels=jnp.asarray(levels,jnp.int32)
        self.num_problems=int(sum(counts));self.num_actions=36+int(self.stop_enabled)
        self.eos_action=self.stop_action
        self.PAD_TOKEN=0;self.BOS_TOKEN=1;self.SEP_TOKEN=2
        self.arrow_token=3;self.positive_token=4;self.negative_token=5
        self.digit_offset=6;self.action_offset=16;self.invalid_token=52;self.stop_token=53
        self.vocab_size=54 if self.stop_enabled else 53
        self.width=max(len(str(abs(self.minimum))),len(str(abs(self.maximum))),len(str(max_steps)))
        self.number_width=self.width+1  # Always sign + decimal digits, never a whole-number token.
        self.prompt_len=2+3*self.number_width+2
        self.record_width=2+2*self.number_width
        self.max_seq_len=self.prompt_len+self.max_steps*self.record_width
        self.powers=jnp.asarray(10**np.arange(self.width-1,-1,-1),jnp.int32)
        digest=hashlib.sha256()
        digest.update(np.asarray(self.starts).tobytes());digest.update(np.asarray(self.targets).tobytes())
        self.config=dict(version=self.VERSION,seed=seed,eval_seed=eval_seed,split=split,gamma=gamma,
                         max_steps=max_steps,number_min_value=self.minimum,number_max_value=self.maximum,
                         number_min_distance=number_min_distance,number_max_distance=number_max_distance,
                         number_teacher_probability=number_teacher_probability,number_split_mode=number_split_mode,number_stop=int(self.stop_enabled),
                         num_pairs=self.num_problems,num_actions=self.num_actions,vocab_size=self.vocab_size,stop_action=self.stop_action,
                         shortest_distance_unit="arithmetic moves; STOP is an additional action when used",
                         operation_names=['add','subtract','multiply','divide_exact'],identity_action=18,
                         distance_counts={str(k):int(v) for k,v in zip(levels,counts)},
                         sampling='uniform distance bucket, then uniform pair within bucket',
                         reward=('1 iff current value equals target on STOP or deadline' if self.stop_enabled else '1 iff final value equals target at exact deadline; otherwise 0'),
                         invalid_action='consume a step, preserve current value',
                         task_bank_hash=digest.hexdigest())

    def encode_number(self,value):
        digits=(jnp.abs(value)//self.powers)%10+self.digit_offset
        return jnp.concatenate([jnp.where(value<0,self.negative_token,self.positive_token)[None],digits])

    def reset_problem(self,index):
        index=jnp.asarray(index,jnp.int32);start=self.starts[index];target=self.targets[index]
        prefix=jnp.concatenate([jnp.array([self.BOS_TOKEN]),self.encode_number(start),
                                jnp.array([self.arrow_token]),self.encode_number(target),jnp.array([self.SEP_TOKEN]),
                                self.encode_number(jnp.array(self.max_steps)),jnp.array([self.SEP_TOKEN])])
        history=jnp.zeros(self.max_seq_len,jnp.int32).at[:self.prompt_len].set(prefix)
        state=NumberState(index,start,target,jnp.array(0,jnp.int32),history,jnp.array(False))
        return history,state

    def reset(self,rng,params=None):
        bk,pk=jax.random.split(rng)
        bucket=jax.random.randint(bk,(),0,len(self.bucket_counts))
        index=self.bucket_offsets[bucket]+jax.random.randint(pk,(),0,self.bucket_counts[bucket])
        return self.reset_problem(index)

    def evaluation_problem_ids(self,count,seed):
        rng=np.random.default_rng(seed);count=min(count,self.num_problems)
        buckets=[rng.permutation(int(n))+int(o) for o,n in zip(self.bucket_offsets,self.bucket_counts)]
        selected=[];positions=[0]*len(buckets)
        while len(selected)<count:
            for i,bucket in enumerate(buckets):
                if positions[i]<len(bucket) and len(selected)<count:
                    selected.append(bucket[positions[i]]);positions[i]+=1
        return np.sort(np.asarray(selected,dtype=np.int32))

    def initial_distance(self,state):
        return self.distances[self.starts[state.prompt_idx]-self.minimum,state.target_value-self.minimum]

    @staticmethod
    def encode_action(operation,digit):return operation*9+digit-1

    def step(self,rng,state,action,params=None):
        del rng,params
        action=jnp.asarray(action,jnp.int32);safe=jnp.clip(action,0,35)
        arithmetic=(action>=0)&(action<36)&self.valid_actions[state.current_value-self.minimum,safe]
        stop=self.stop_enabled&(action==36)
        valid=arithmetic|stop
        value=jnp.where(arithmetic,self.next_values[state.current_value-self.minimum,safe],state.current_value)
        count=state.step_count+1;done=(count>=self.max_steps)|stop
        token=jnp.where(stop,self.stop_token,self.action_offset+safe)
        record=jnp.concatenate([jnp.array([token,jnp.where(valid,self.SEP_TOKEN,self.invalid_token)]),
                                self.encode_number(value),self.encode_number(jnp.maximum(self.max_steps-count,0))])
        offset=self.prompt_len+jnp.minimum(state.step_count,self.max_steps-1)*self.record_width
        history=jax.lax.dynamic_update_slice(state.history_tokens,record,(offset,))
        new=NumberState(state.prompt_idx,value,state.target_value,count,history,done)
        new=jax.tree_util.tree_map(lambda x,y:jnp.where(state.done,y,x),new,state)
        reward=(~state.done&done&(value==state.target_value)).astype(jnp.float32)
        return new.history_tokens,new,reward,new.done,dict(invalid_action=(~state.done)&~valid)

    def reference_action(self,state):
        following=self.next_values[state.current_value-self.minimum]
        d=self.distances[following-self.minimum,state.target_value-self.minimum]
        # Exact search is used only for demonstrations/reference-policy actions.
        # It is not a value oracle for a history-dependent learned policy.
        best=jnp.argmin(jnp.where(self.valid_actions[state.current_value-self.minimum],d,65535))
        return jnp.where(state.current_value==state.target_value,36 if self.stop_enabled else 18,best).astype(jnp.int32)

    def action_probabilities(self,state):
        teacher=jax.nn.one_hot(self.reference_action(state),self.num_actions)
        return self.teacher_probability*teacher+(1-self.teacher_probability)*jnp.ones(self.num_actions)/self.num_actions

    def sample_base_action(self,rng,state,params=None):
        return jax.random.categorical(rng,jnp.log(self.action_probabilities(state))).astype(jnp.int32)

    def export_tasks(self,path):
        path=Path(path)
        depth=np.repeat(np.asarray(self.levels),np.asarray(self.bucket_counts))
        np.savez_compressed(path,start=np.asarray(self.starts),target=np.asarray(self.targets),shortest_steps=depth)
