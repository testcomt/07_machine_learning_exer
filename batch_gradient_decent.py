#! /usr/local/bin/python3
# -*- coding: utf-8 -*-

import math

training_set = []
NUMBER_OF_FEATURES = 1
init_weights = ()
LEARNING_RATE = -0.000000001


def hypothesis_value(weights, x)->float:
    """cal h for one training element
    :param *x, feature values for one element
    """
    if len(weights) != len(x) + 1:
        raise ValueError
    x.insert(0, 1)
    h = 0
    for weight in weights:
        for value in x:
            h += weight * value
    return h


def one_step_weight_update(current_weights, jth_feature)->float:
    """Batch Gradient Dscent: For every update of weight_j, use all records
    :param jth_feature - including 0th non-feature
    j = 0 means para0, corresponding to no feature
    :param current_weights - current weights for all features, including feature 0
    """

    sum_of_errors = 0

    for ith_element in training_set:

        x_j = ith_element[jth_feature - 1] if jth_feature else 1
        h = hypothesis_value(current_weights, ith_element[:-1])
        sum_of_errors += x_j * (h - ith_element[-1])

    return current_weights[jth_feature] + LEARNING_RATE * sum_of_errors


def cost_func(weights)->float:
    """cal cost based on current weights
    :param weights - current weights of each feature, including feature 0
    """

    cost = 0
    for ith_element in training_set:
        cost += math.pow(hypothesis_value(weights, ith_element[:-1]) - ith_element[-1], 2)
    return cost / 2


def next_cost_calc(prev_weights):
    """calc next iteration of cost function based on previous iteration weights"""

    next_weights = []
    for j in range(NUMBER_OF_FEATURES + 1):
        next_weights.append(one_step_weight_update(prev_weights, j))
    next_cost = cost_func(next_weights)
    return next_cost, next_weights


def convergence_check():

    prev_weights = init_weights
    prev_cost = cost_func(prev_weights)

    next_cost, next_weights = next_cost_calc(prev_weights)
    print(next_cost)

    cycle = 0
    while next_cost < prev_cost:
        prev_cost = next_cost
        prev_weights = next_weights
        next_cost, next_weights = next_cost_calc(prev_weights)
        cycle += 1
        print("cycles = ", cycle)
        print("next cost = %f, next weights are %s", (next_cost, next_weights))
    else:
        print("\n\nCost is ", prev_cost)
        print("weights are", prev_weights)


def test_one_feature():

    # [living area, price]

    global training_set
    training_set = [[2104, 400],
                    [1416, 232],
                    [1534, 315],
                    [852, 178],
                    [1940, 240]]

    global NUMBER_OF_FEATURES
    NUMBER_OF_FEATURES = 1

    global init_weights
    init_weights = (0, 0)

    global LEARNING_RATE
    LEARNING_RATE = -0.000000001

    convergence_check()


def test_two_features():
    """one feature is living area
    the other feature is square of living area
    """

    # [living area, square of living area, price]

    global training_set
    training_set = [[2104, 2104*2104, 400],
                    [1416, 1416*1416, 232],
                    [1534, 1534*1534, 315],
                    [852, 852*852, 178],
                    [1940, 1940*1940, 240]]

    global NUMBER_OF_FEATURES
    NUMBER_OF_FEATURES = 2

    global init_weights
    init_weights = (0, 0, 0)

    # Cost is  17821.23666669045, when learning_rate is -0.1e-13 (least value)
    # weights are [7.899092315128317e-11, 8.495361206780859e-08, 9.131590067807478e-05]
    global LEARNING_RATE
    LEARNING_RATE = -0.1e-13

    convergence_check()


if __name__ == '__main__':
    # test_one_feature()
    test_two_features()


# 1. LEARNING_RATE IS SET TWO DIFF VALUES, SEE CONVERGENCE SPEED
# 2. TODO: LEARNING_RATE is now human-set value. Can it be auto-adjusted by program?
# 3. TODO: draw a graph to show the changing state
# 4. TODO: import data from outside (large files, website, etc)
# 5. TODO: factors influencing cost function? why least cost is 6974 rather than 0?
# init_weights = (0, 0)
#
# LEARNING_RATE = -0.00000000001
#
# cycles =  109365
# next cost = %f, next weights are %s 6974.022381752115 [0.00012601277786493533, 0.17082227521785773]
#
#
# Cost is  6974.022381752115
# weights are [0.000126012549015025, 0.17082227543565237]
#
# /////////////////////////////////////
#
# init_weights = (0, 0)
#
# LEARNING_RATE = -0.000000001
#
#
# cycles =  1278
# next cost = %f, next weights are %s (6974.022381706409, [0.00013025420645806495, 0.1708181106353402])
#
#
# Cost is  6974.022381706409
# weights are [0.00013023132206976893, 0.17081813343451907]
