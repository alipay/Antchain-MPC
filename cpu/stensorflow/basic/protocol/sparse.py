#!/usr/bin/env python
# coding=utf-8
"""
   Alipay.com Inc.
   Copyright (c) 2004-2022 All Rights Reserved.
   ------------------------------------------------------
   File Name : sparse
   Author : qizhi.zqz
   Email: qizhi.zqz@antgroup.com
   Create Time : 2022/9/19 下午2:21
   Description : description what the main function of this file
"""

# import dgl
# dgl.backend.set_default_backend('tensorflow')
# dataset = dgl.data.CoraGraphDataset()
#
# g = dataset[0]
# print_ops = []
# A = g.adjacency_matrix().indices.numpy()
#
# print(A)
# m = g.num_nodes()
# n = m
# r = g.num_edges()


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#






import tensorflow as tf
from stensorflow.global_var import StfConfig
from stensorflow.engine.start_server import start_local_server
from stensorflow.random.random import gen_rint64, gen_rint64_from_seed
tf.compat.v1.disable_eager_execution()
from stensorflow.basic.basic_class.private import PrivateTensor, get_device
from stensorflow.basic.basic_class.pair import SharedPair
from stensorflow.basic.basic_class.share import SharedTensor
from stensorflow.basic.basic_class.base import SharedTensorBase
from stensorflow.basic.operator.algebra import concat
from stensorflow.basic.operator.selectshare import select_share
import numpy as np
from typing import Union

print_ops = []


def get_sparseP(P, m=None):
    """

    :param P: [2,2,3,4]
    :param m:
    :param n:
    :return: Sparse Tensor : [(2,0), (2,1), (3,2), (4,3)]
    """
    r = tf.shape(P)[0]
    if m is None:
        m = r
    indices_P = tf.concat([tf.reshape(P, [-1, 1]), tf.expand_dims(tf.range(start=0, limit=P.shape[0]), axis=1)],
                            axis=1)
    indices_P = tf.cast(indices_P, 'int64')
    sparse_P = tf.sparse.SparseTensor(indices=indices_P, values=tf.ones(shape=[r], dtype='int32'), dense_shape=[m, r])
    return sparse_P


def get_sparseQ(Q, n=None):
    """

    :param Q: [2,2,3,4]
    :param m:
    :param n:
    :return: Sparse Tensor : [(0,2), (1,2), (2,3), (3,4)]
    """
    r = tf.shape(Q)[0]
    if n is None:
        n = r
    indices_Q = tf.concat([tf.expand_dims(tf.range(start=0, limit=Q.shape[0]), axis=1), tf.reshape(Q, [-1, 1])],
                            axis=1)
    indices_Q = tf.cast(indices_Q, 'int64')
    sparse_Q = tf.sparse.SparseTensor(indices=indices_Q, values=tf.ones(shape=[r], dtype='int32'), dense_shape=[r, n])
    return sparse_Q


class Permutations:
    def __init__(self, s):
        self.s = tf.cast(s, 'int32')
        # for example, [2, 3, 1, 0] means
        # [[0, 1, 2, 3],
        # [2, 3, 1, 0]]

    def __mul__(self, other):
        if isinstance(other, Permutations):
            s = tf.gather(self.s, other.s)
            return Permutations(s)
        else:
            raise Exception("must have isinstance(other, Permutations)")
    def __invert__(self):
        s = tf.argsort(self.s)
        return Permutations(s)

    def random_uniform_adjoint(self, seed):
        # print("l126, self.s.shape=", self.s.shape)
        range_ = tf.range(self.s.shape[0])
        s = tf.random.shuffle(range_, seed=seed)
        return Permutations(s)


    def act(self, x: Union[tf.Tensor, SharedTensor]):
        """

        :param x:
        :return:  [2, 3, 1, 0].act(x) == sparsematrix([[2,0], [3,1], [1,2], [0,3]]) @ x
        """
        if isinstance(x, SharedTensorBase):
            x1 = x.inner_value
        else:
            x1 = x

        inv = self.__invert__().s
        y1 = tf.gather(x1, inv, axis=0)
        if isinstance(x, SharedTensorBase):
            y = SharedTensor(inner_value=y1, module=x.module)
        else:
            y = y1
        return y

    def to_sparse_matrix(self):
        return get_sparseP(self.s)




class PermutationAndShift:
    def __init__(self, s: Union[tf.Tensor, Permutations], a: Union[tf.Tensor, SharedTensor]):
        """

        :param s:  tf.Tensor of dtype=int32
        :param a:  tf.Tensor of dtype=int64
        """
        if isinstance(s, Permutations):
            self.s = s
        else:
            self.s = Permutations(s)

        if isinstance(s, SharedTensor):
            self.a = a
        else:
            self.a = SharedTensor(inner_value=a)
        # for example, [2, 3, 1, 0] means
        # [[0, 1, 2, 3],
        # [2, 3, 1, 0]]

    def __mul__(self, other):
        if isinstance(other, PermutationAndShift):
            s = self.s * other.s
            a = self.s.act(other.a)+self.a
            return PermutationAndShift(s, a)
        else:
            raise Exception("must have isinstance(other, Permutations)")

    def __invert__(self):
        s = ~self.s
        a = -(~s).act(self.a)
        return PermutationAndShift(s, a)

    def random_uniform_adjoint(self, seed):
        s = self.s.random_uniform_adjoint(seed)
        a = self.a.random_uniform_adjoint(seed)
        return PermutationAndShift(s, a)

    def act(self, x: Union[tf.Tensor, SharedTensor]):
        """

        :param x:
        :return:  [2, 3, 1, 0].act(x) == sparsematrix([[2,0], [3,1], [1,2], [0,3]]) @ x
        """

        # inv = self.__invert__().s
        # y1 = tf.gather(x1, inv, axis=0)
        y = self.s.act(x)+self.a
        # if isinstance(x, SharedTensorBase):
        #     y = SharedTensor(inner_value=y1, module=x.module)
        # else:
        #     y = y1
        return y



def diff(x: SharedPair):
    d = x.shape[0]
    z = concat([x[0:1, ...], (x[1:d, ...] - x[0:(d - 1), ...])], axis=0)
    z = SharedPair.from_SharedPairBase(z, x.op_map)
    return z

def integrate(x: SharedPair, reverse=False):
    # z=tf.math.cumsum(x, axis=0)
    # print("l 163 x=", x)
    z = x.cumulative_sum(axis=0, reverse=reverse)
    return z


class PrivatePermutations:
    def __init__(self, tf_s: tf.Tensor, owner, n=None):
        self.owner = get_device(owner)
        with tf.device(self.owner):
            if n is not None:
                tf_s = tf.reshape(tf_s, [n])
            self.s = Permutations(tf_s)
            # for example, [2, 3, 1, 0] means
            # [[0, 1, 2, 3],
            # [2, 3, 1, 0]]

    def __mul__(self, other):
        if isinstance(other, PrivatePermutations):
            if self.owner == other.owner:
                with tf.device(self.owner):
                    s = self.s * other.s
                    return PrivatePermutations(s.s, self.owner)
            else:
                raise Exception("must have self.owner == other.owner")
        else:
            raise Exception("must have isinstance(other, Permutations)")

    def __invert__(self):
        with tf.device(self.owner):
            return PrivatePermutations(tf_s=(~self.s).s, owner=self.owner)


    def act_Private(self, x: PrivateTensor):
        """

        :param x:
        :return:  [2, 3, 1, 0].act(x) == sparsematrix([[2,0], [3,1], [1,2], [0,3]]) @ x
        """
        if self.owner == x.owner:
            with tf.device(self.owner):
                inner_value = self.s.act(x.inner_value)
            return PrivateTensor(owner=self.owner, fixedpoint=x.fixedpoint,
                                 inner_value=inner_value, module=x.module, op_map=x.op_map)
        else:
            seed_s = None
            seed_x = None
            seed_u = None
            with tf.device(StfConfig.RS[0]):
                s_adjoint = self.s.random_uniform_adjoint(seed=seed_s)
                x_adjoint = x.to_SharedTensor().random_uniform_adjoint(seed=seed_x)
                u = s_adjoint.act(x_adjoint)
                #uL = u.random_uniform_adjoint(seed=seed_u)
                uL = u.random_uniform_adjoint(seed=seed_u)
                uR = u - uL
            with tf.device(x.owner):
                dx = x.to_SharedTensor()-x_adjoint
            with tf.device(self.owner):
                ds = self.s * (~s_adjoint)
                yL = self.s.act(dx) + ds.act(uL)
            with tf.device(x.owner):
                yR = ds.act(uR)
            return SharedPair(ownerL=self.owner, ownerR=x.owner, xL=yL, xR=yR, fixedpoint=x.fixedpoint,
                              shape=x.shape, op_map=x.op_map)

    def act_SharedPair(self, x: SharedPair):
        if self.owner == x.ownerL:
            seed_s = None
            seed_x = None
            seed_u = None
            with tf.device(StfConfig.RS[0]):
                s_adjoint = self.s.random_uniform_adjoint(seed=seed_s)
                x_adjoint = x.xL.random_uniform_adjoint(seed=seed_x)
                u = s_adjoint.act(x_adjoint)
                uL = u.random_uniform_adjoint(seed=seed_u)
                uR = u - uL
            with tf.device(self.owner):
                delta_s = self.s * (~s_adjoint)
            with tf.device(x.ownerR):
                delta_x = x.xR - x_adjoint
                yR = delta_s.act(uR)
            with tf.device(self.owner):
                yL = self.s.act(x.xL+delta_x) +delta_s.act(uL)
            return SharedPair(ownerL=x.ownerL, ownerR=x.ownerR, xL=yL, xR=yR,
                              fixedpoint=x.fixedpoint, op_map=x.op_map)
        elif self.owner == x.ownerR:
            return self.act_SharedPair(x.mirror())
        else:
            raise Exception("must have self.owner==x.ownerL or self.owner==x.ownerR")

    def act(self, x: Union[PrivateTensor, SharedPair]):
        if isinstance(x, PrivateTensor):
            return self.act_Private(x)
        elif isinstance(x, SharedPair):
            return self.act_SharedPair(x)
        else:
            raise Exception("must have isinstance(x, PrivateTensor) or isinstance(x, SharedPair) but x=", x)




def decompose_row_column(A):
    """
    :param A:  mxn sparse 0-1 matrix represented by [ [row0, col0], [row1, col1], ..... ]
    :return: (P, sigma, Q) s.t A=P sigma Q^T,
    where P mxr sparse 0-1 matrix, there is only 1 in every col, and the row-index is non-decrease, represent by [row0, row1, ...]
          Q  nxr sparse 0-1 matrix, there is only 1 in every row, and the row-index is non-decrease, represent by [row0, row1, ...]
          sigma in S_r   represent by [sigma(0), sigma(1), ...]
    """
    P = A[:, 0]
    Q = A[:, 1]
    sigma = tf.argsort(Q)
    sigma = Permutations(sigma)
    inv_sigma = ~sigma
    Q = inv_sigma.act(Q)

    return P, sigma, Q

def decompose_P(P, m, r):
    """

    :param P:  [0,0,0,0,2,2,2,4,4,4,4,4]
            =
               [[ 1,1,1,1,0,0,0,0,0,0,0],
                [ 0,0,0,0,0,0,0,0,0,0,0],
                [ 0,0,0,0,1,1,1,0,0,0,0],
                [ 0,0,0,0,0,0,0,0,0,0,0],
                [ 0,0,0,0,0,0,0,1,1,1,1]]
    :return:   s2=[1->0, 3->1,     0->2 ,2->3 ,4->4] in S_m
               K=[(2,9), (3,10), (4,11)]
               s1 = [ 3->9, 6->10, 11->11,        0->0,1->1,2->2,4->3,5->4,7->5,8->6,9->7,10->8]  in Sr
    """

    s2_1, idx = tf.unique(P) # back part of inv_s2.  s2_1=[0,2,4], idx=[0,0,0,0,1,1,1,2,2,2,2,2]
    k = tf.shape(s2_1)[0]
    s2_0 = tf.sets.difference(tf.reshape(tf.range(start=0, limit=m, dtype='int64'), [1,-1]), tf.reshape(s2_1, [1,-1])) # front part
    # [1, 3]
    # print("s2_0=", s2_0)
    s2_0 = tf.sparse.to_dense(s2_0)
    # print("s2_0=", s2_0)
    s2_0 = tf.reshape(s2_0, [-1])
    # print("s2_0=", s2_0)
    s2 = tf.concat([s2_0, s2_1], axis=0)  # [1,3, 0,2,4]
    s2 = Permutations(s2)
    inv_diff_idx = tf.concat([idx[1:]-idx[0:-1], [1]], axis=0)
    # print("inv_diff_idx=", inv_diff_idx)
    # [0,0,0,1,0,0,1,0,0,0,0,1]
    idx1 = tf.where(tf.equal(inv_diff_idx, 1))
    #print_op = tf.print(inv_diff_idx, idx1)
    idx1 = tf.squeeze(idx1)  # [3,6,11]
    idx1 = tf.cast(idx1, 'int32')
    # print("idx1=", idx1)
    # print_op = tf.print("l224, idx1=", idx1)
    # print_ops.append(print_op)
    idx0 = tf.sets.difference(tf.reshape(tf.range(start=0, limit=r), [1,-1]), tf.reshape(idx1, [1,-1])) #[0,1,2,4,5,7,8,9,10]
    # print("idx0=", idx0)
    # print_op = tf.print("l228, idx0=", idx0)
    # print_ops.append(print_op)
    idx0 = tf.sparse.to_dense(idx0)
    idx0 = tf.reshape(idx0, [-1])
    # print("idx0=", idx0)
    # print_op = tf.print("l233, idx0=", idx0)
    # print_ops.append(print_op)

    inv_s1 = tf.concat([idx0, idx1], axis=0) # [0,1,2,4,5,7,8,9,10,    3,6,11]
    inv_s1 = Permutations(inv_s1)
    s1 = ~inv_s1

    return s2, k, s1



def decompose_Q(Q, r, n):
    """
    :param Q:  [0,0,2,2,2,4,4,5] =
               [1,0,0,0,0,0,0],
               [1,0,0,0,0,0,0],
               [0,0,1,0,0,0,0],
               [0,0,1,0,0,0,0],
               [0,0,1,0,0,0,0],
               [0,0,0,0,1,0,0],
               [0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0]
    :param r:  8
    :param n:  7
    :return:   s1=[0->0, 2->1, 4->2, 5->3;      1->4, 3->5, 6->6] in Sn,
    k=4,
    s2 = [0->0, 2->1, 5->2, 7->3;  1->4, 3->5, 4->6, 6->7] in Sr

    """
    inv_s1_0, idx1 = tf.unique(Q) # front part of inv_s1; inv_s1_0 = [0,2,4,5],  idx1=[0,0,1,1,1,2,2,3]
    k = tf.shape(inv_s1_0)[0]
    inv_s1_1 = tf.sets.difference(tf.reshape(tf.range(start=0, limit=n, dtype='int64'), [1, -1]),
                                  tf.reshape(inv_s1_0, [1, -1]))  # back part
    inv_s1_1 = tf.sparse.to_dense(inv_s1_1)
    inv_s1_1 = tf.reshape(inv_s1_1, [-1])
    inv_s1 = tf.concat([inv_s1_0, inv_s1_1], axis=0)  #
    inv_s1 = Permutations(inv_s1)
    s1 = ~inv_s1

    diff_idx1 = idx1 - tf.concat([[-1], idx1[0:-1]], axis=0)    # [1,0,1,0,0,1,0,1]
    # print_op = tf.print("diff_idx1=", diff_idx1)
    # print_ops.append(print_op)
    s2_0 = tf.where(tf.equal(diff_idx1, 1))  # [0,2,5,7]
    s2_0 = tf.cast(s2_0, 'int32')
    # print("s2_0=", s2_0)
    s2_1 = tf.sets.difference(tf.reshape(tf.range(start=0, limit=r), [1,-1]), tf.reshape(s2_0, [1,-1]))
    s2_1 = tf.sparse.to_dense(s2_1)
    s2_1 = tf.reshape(s2_1, [-1])
    # print("s2_1=", s2_1)
    s2_0 = tf.reshape(s2_0, [-1])
    s2 = tf.concat([s2_0, s2_1], axis=0)
    # print_op = tf.print("s2=", s2)
    # print_ops.append(print_op)
    s2 = Permutations(s2)
    return s2, k, s1






def P_mult(s5: PrivatePermutations, kP, s4: PrivatePermutations, x3: SharedPair, m: int):
    x4 = integrate(x3)
    #print_op = tf.print("l450, x4=", x4)
    #print_ops.append(print_op)
    y4 = s4.act(x4)
    #print_op = tf.print("l453, y4=", y4)
    #print_ops.append(print_op)


    with tf.device(s4.owner):
        IkP = tf.concat([tf.zeros([m - kP, 1], dtype='int64'), tf.ones([kP, 1], dtype='int64')], axis=0)
        IkP = PrivateTensor(owner=s4.owner, inner_value=IkP, module=2, fixedpoint=0)

    # x5_back = y4[-m:, ...]
    # x5_front_shape = x5_back.shape.as_list()
    # x5_front_shape[0] = m - kP
    # x5_front = tf.zeros(x5_front_shape, dtype='int64')
    # x5_front = SharedPair(ownerL=x3.ownerL, ownerR=x3.ownerR)
    # #x5_back = tf.cast(x5_back, dtype='float64')
    #
    # x5 = tf.concat([x5_front, x5_back], axis=0)
    x5 = select_share(IkP, y4[-m:, ...])
    #print_op = tf.print("l460, x5=", x5)
    #print_ops.append(print_op)
    y5 = diff(x5)
    #print_op = tf.print("l463 y5=", y5)
    #print_ops.append(print_op)
    z5 = s5.act(y5)
    return z5


def Q_mult(s2: PrivatePermutations, kQ, s1: PrivatePermutations, r, x: Union[PrivateTensor, SharedPair])->SharedPair:
    # print("l334, x=", x)
    n = x.shape[0]
    y1 = s1.act(x)
    # print_op = tf.print("l399 s1=", s1.s, "x=", x, "y1=", y1)
    # print_ops.append(print_op)
    # print("l341, y1=", y1)
    z1 = diff(y1)
    # print("l343 z1=", z1)

    with tf.device(s1.owner):
        IkQ = tf.concat([tf.ones([kQ, 1], dtype='int64'), tf.zeros([r - kQ, 1], dtype='int64')], axis=0)
        IkQ = PrivateTensor(owner=s1.owner, inner_value=IkQ, module=2, fixedpoint=0)
        ext = tf.zeros([r - n] + z1.shape[1:], dtype='int64')
        ext = PrivateTensor(owner=s1.owner, inner_value=ext, module=None, fixedpoint=0)
    x2 = concat([z1, ext], axis=0)
    x2 = select_share(IkQ, x2)
    y2 = s2.act(x2)
    #print_op = tf.print("l418 x2=", x2, "s2=", s2.s, "y2=", y2)
    #print_ops.append(print_op)
    x3 = integrate(y2)
    return x3



# def Qt_mult(s2: PrivatePermutations, kQ, s1: PrivatePermutations, r, x3d: Union[PrivateTensor, SharedPair])->SharedPair:
#     y2d = integrate(x3d, reverse=True)
#     x2d = (~s2).act(y2d)




class PrivateSparseMatrix:
    def __init__(self, A: tf.Tensor, m, n, r, owner):
        self.A = A
        self.r = r
        self.n = n
        self.m = m
        self.owner = get_device(owner)
        with tf.device(self.owner):
            values = tf.ones_like(A[:, 0], dtype='int64')
            self.sparse_A = tf.sparse.SparseTensor(indices=A, values=values, dense_shape=[m, n])
            P, s3, Q = decompose_row_column(A)
            s5, self.kP, s4 = decompose_P(P, m, r)
            s2, self.kQ, s1 = decompose_Q(Q, r, n)
            self.s1 = PrivatePermutations(tf_s=s1.s, owner=owner, n=self.n)
            self.s2 = PrivatePermutations(tf_s=s2.s, owner=owner, n=self.r)
            self.s3 = PrivatePermutations(tf_s=s3.s, owner=owner, n=self.r)
            self.s4 = PrivatePermutations(tf_s=s4.s, owner=owner, n=self.r)
            self.s5 = PrivatePermutations(tf_s=s5.s, owner=owner, n=self.m)


    def act(self, x: Union[PrivateTensor, SharedPair]):
        # if norm == 'both':
        #     x = x * invers_sqrt(self.in_degrees)
        # elif norm == 'right':
        #     x /= self.in_degrees
        x3 = Q_mult(self.s2, self.kQ, self.s1, self.r, x)
        x3 = self.s3.act(x3)
        z5 = P_mult(self.s5, self.kP, self.s4, x3, self.m)
        # if norm == 'both':
        #     z5 = z5 * invers_sqrt(self.out_degrees)
        return z5

    def transpose(self):
        A = tf.concat([self.A[:,0:1], self.A[:,1:2]], axis=1)
        return PrivateSparseMatrix(A, m=self.n, n=self.m, r=self.r, owner=self.owner)

# if __name__ == '__main__':
#     start_local_server(config_file="../../../conf/config.json")
#     x_np = np.random.random(size=[n,3])
#     #tes_matrix_multi(A, x)
#     x = PrivateTensor(owner="R")
#     x.load_from_numpy(x_np)
#
#     A = PrivateSparseMatrix(A, m, n, r, "L")
#
#     y = A.act(x)
#
#     print("y=", y)
#
#     ## -----------------------------------val------------------------------
#     A = tf.cast(A.A, 'int64')
#     values = tf.ones_like(A[:,0], dtype='float64')
#     sparse_A = tf.sparse.SparseTensor(indices=A, values=values, dense_shape=[m,n])
#     print("sparse_A=", sparse_A)
#     print("l453, x=", x)
#     z = tf.sparse.sparse_dense_matmul(sparse_A, tf.cast(x.to_tf_tensor("R"), 'float64'))
#     ## -----------------------------------val------------------------------
#
#
#
#     sess = tf.compat.v1.Session(target=StfConfig.target)
#     sess.run(tf.compat.v1.initialize_all_variables())
#     #ymz = sess.run(y.to_tf_tensor("R")-z)
#     #print(ymz)
#
#     y = sess.run(y.to_tf_tensor("R"))
#     print(y)



