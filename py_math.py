import numpy as np


class PyMath:
    @staticmethod
    def linear_zero(y, slope, intercept):
        return (y - intercept) / slope

    @staticmethod
    def calc_polynomial(coefs, powers, value):
        total = 0
        for i in range(len(coefs)):
            total += (value**powers[i]) * coefs[i]
        return total

    @staticmethod
    def poly_derivative(coefs, powers):
        new_coefs = []
        new_powers = []
        for base in range(len(coefs)):
            if powers[base] != 0:
                new_coefs.append(coefs[base] * powers[base])
                new_powers.append(powers[base] - 1)
        return new_coefs, new_powers

    @staticmethod
    def multiply_poly(coefs_1, powers_1, coefs_2, powers_2):
        new_coefs = []
        new_powers = []
        for c1 in range(len(coefs_1)):
            for c2 in range(len(coefs_2)):
                new_coefs.append(coefs_1[c1] * coefs_2[c2])
                new_powers.append(powers_1[c1] + powers_2[c2])
        simp_coefs, simp_powers = PyMath.poly_simplify(new_coefs, new_powers)
        return simp_coefs, simp_powers

    @staticmethod
    def poly_simplify(coefs, powers):
        simp_coefs = []
        simp_powers = []
        excluded_indices = []
        changes = -1
        while changes != 0:
            changes = 1
            simp_coefs = []
            simp_powers = []

            excluded_indices = []
            for p1 in range(len(powers)):
                if p1 not in excluded_indices:
                    duplicate_coefs = [coefs[p1]]
                    for p2 in range(p1 + 1, len(powers)):
                        if powers[p1] == powers[p2]:
                            excluded_indices.append(p2)
                            duplicate_coefs.append(coefs[p2])
                            changes += 1

                    simp_powers.append(powers[p1])
                    simp_coefs.append(sum(duplicate_coefs))
            powers = simp_powers
            coefs = simp_coefs
            changes -= 1
        c = 0
        while c < len(simp_coefs):
            if simp_coefs[c] == 0:
                del simp_coefs[c]
                del simp_powers[c]
            else:
                c += 1
        if len(simp_coefs) != 0:
            simp_powers, simp_coefs = [list(x) for x in zip(*sorted(zip(simp_powers, simp_coefs), reverse=True))]
        return simp_coefs, simp_powers

    @staticmethod
    def poly_add(coefs1, powers1, coefs2, powers2):
        new_coefs = []
        new_powers = []
        for p1 in range(len(powers1)):
            total = coefs1[p1]
            for p2 in range(len(powers2)):
                if powers1[p1] == powers2[p2]:
                    total += coefs2[p2]
            new_coefs.append(total)
            new_powers.append(powers1[p1])

        for p in range(len(powers2)):
            if powers2[p] not in new_powers:
                new_coefs.append(coefs2[p])
                new_powers.append(powers2[p])

        sorted_coefs, sorted_powers = PyMath.poly_simplify(new_coefs, new_powers)
        return sorted_coefs, sorted_powers

    @staticmethod
    def poly_subtract(coefs1, powers1, coefs2, powers2):
        coefs1, powers1 = PyMath.poly_simplify(coefs1, powers1)
        coefs2, powers2 = PyMath.poly_simplify(coefs2, powers2)
        new_coefs = []
        new_powers = []
        for p1 in range(len(powers1)):
            total = coefs1[p1]
            for p2 in range(len(powers2)):
                if powers1[p1] == powers2[p2]:
                    total -= coefs2[p2]
            new_coefs.append(total)
            new_powers.append(powers1[p1])

        for p in range(len(powers2)):
            if powers2[p] not in new_powers:
                new_coefs.append(-coefs2[p])
                new_powers.append(powers2[p])

        sorted_coefs, sorted_powers = PyMath.poly_simplify(new_coefs, new_powers)
        return sorted_coefs, sorted_powers

    @staticmethod
    def poly_to_string(coefs, powers):
        coefs, powers = PyMath.poly_simplify(coefs, powers)
        result = ""
        for x in range(len(coefs)):
            if powers[x] != 0 and powers[x] != 1:
                if x != len(coefs) - 1:
                    result += str(coefs[x]) + "λ^" + str(powers[x]) + " + "
                else:
                    result += str(coefs[x]) + "λ^" + str(powers[x])
            elif powers[x] == 1:
                if x != len(coefs) - 1:
                    result += str(coefs[x]) + "λ" + " + "
                else:
                    result += str(coefs[x]) + "λ"
            else:
                result += str(coefs[x])
        return result

    @staticmethod
    def poly_zeros(coefs, powers):
        ranges = PyMath.intermediate_value(coefs, powers, max(powers))
        results = PyMath.newton_method(coefs, powers, ranges)
        temp_results = []
        for r in results:
            test_coefs = coefs
            test_powers = powers
            iterating = True
            while iterating:
                if abs(0 - PyMath.calc_polynomial(test_coefs, test_powers, r)) < 2:
                    temp_results.append(r)
                    test_coefs, test_powers = PyMath.poly_derivative(test_coefs, test_powers)
                else:
                    iterating = False
        return np.array(temp_results)

    @staticmethod
    def intermediate_value(coefs, powers, num_zeros):
        ranges = [[-100, 100]]
        result_real = PyMath.__intermediate_value_recursive(coefs, powers, num_zeros, ranges, 0)
        new_result = []
        for i in result_real:
            new_result.append(i[0])
            new_result.append(i[1])
            new_result.append((i[1] + i[0]) / 2)
        new_result = np.unique(np.array(new_result))
        return new_result

    @staticmethod
    def __intermediate_value_recursive(coefs, powers, num_zeros, cur_ranges, iterations):
        new_ranges = []
        for r in cur_ranges:
            new_ranges.append([r[0], r[1] - (r[1] - r[0]) / 2])
            new_ranges.append([r[1] - ((r[1] - r[0]) / 2), r[1]])
        test_ranges = []
        for r in new_ranges:
            if PyMath.calc_polynomial(coefs, powers, r[0]) * PyMath.calc_polynomial(coefs, powers, r[1]) <= 0:
                test_ranges.append(r)
        if len(test_ranges) >= num_zeros or iterations > num_zeros * 2:
            return test_ranges

        iterations += 1
        return PyMath.__intermediate_value_recursive(coefs, powers, num_zeros, new_ranges, iterations)

    @staticmethod
    def newton_method(coefs, powers, starting_points):
        zeros = []
        for i in starting_points:
            real_val = PyMath.__newton_method_recursive(coefs, powers, i, 0)
            if real_val is not None and abs(0 - PyMath.calc_polynomial(coefs, powers, real_val)) < 0.01:
                zeros.append(round(real_val, 5))

        result = np.unique(np.array(zeros))
        th = 0.1
        result = np.delete(result, np.argwhere(np.ediff1d(result) <= th) + 1)
        return result

    @staticmethod
    def __newton_method_recursive(coefs, powers, starting_point, estimate_count):
        if estimate_count >= 990:
            return starting_point

        deriv_coef, deriv_powers = PyMath.poly_derivative(coefs, powers)
        function_value = PyMath.calc_polynomial(coefs, powers, starting_point)
        deriv_value = PyMath.calc_polynomial(deriv_coef, deriv_powers, starting_point)
        if deriv_value != 0:
            new_starting = starting_point + (-1 * function_value / deriv_value)
        else:
            return None
        estimate_count += 1
        return PyMath.__newton_method_recursive(coefs, powers, new_starting, estimate_count)

    @staticmethod
    def new_identity(rows):
        i_matrix = np.zeros([rows, rows], float)
        for i in range(len(i_matrix)):
            i_matrix[i][i] = 1

        return i_matrix

    @staticmethod
    def add_matrices(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Matrices not of equal size")

        for row in range(len(matrix1)):
            for column in range(len(matrix2)):
                matrix1[row][column] += matrix2[row][column]

        return matrix1

    @staticmethod
    def matrix_scalar(scalar, matrix):
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                matrix[row][col] *= scalar

        return matrix

    @staticmethod
    def matrices_multiplier(matrix1, matrix2):
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Invalid matrix factor sizes: " + str(len(matrix1[0])) + " != " + str(len(matrix2)))

        new_matrix = np.zeros([len(matrix1), len(matrix2[0])], float)
        for row in range(len(new_matrix)):
            for column in range(len(new_matrix[row])):
                # for every new_matrix position
                for col_val in range(len(matrix1[row])):
                    new_matrix[row][column] += matrix1[row][col_val] * matrix2[col_val][column]

        return new_matrix

    @staticmethod
    def combine_matrices(matrix1, matrix2):
        if len(matrix1) != len(matrix2):
            raise ValueError("Invalid combination for given matrix sizes")
        new_array = np.empty((len(matrix1), len(matrix1[0]) + len(matrix2[0])), float)
        for row in range(len(new_array)):
            for column in range(len(new_array[row])):
                if column < len(matrix1[0]):
                    new_array[row][column] = matrix1[row][column]
                else:
                    new_array[row][column] = matrix2[row][column - len(matrix1[0])]
        return new_array

    @staticmethod
    def matrix_determinant(matrix):
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix rows and columns not equal size")

        return PyMath.__matrix_determinant_recursive(matrix)

    @staticmethod
    def __matrix_determinant_recursive(matrix):
        if len(matrix) == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

        new_total = 0
        sign = 1
        for i in range(len(matrix[0])):
            new_matrix = []
            if i == 0:
                for row in range(len(matrix) - 1):
                    new_row = [matrix[row + 1][x + 1] for x in range(len(matrix) - 1)]
                    new_matrix.append(new_row)
            elif i == len(matrix[0]) - 1:
                for row in range(len(matrix) - 1):
                    new_row = [matrix[row + 1][x] for x in range(len(matrix) - 1)]
                    new_matrix.append(new_row)
            else:
                for row in range(len(matrix) - 1):
                    new_row = []
                    for column in range(len(matrix[row + 1])):
                        if column != i:
                            new_row.append(matrix[row + 1][column])
                    new_matrix.append(new_row)

            new_matrix = np.array(new_matrix)
            new_total += sign * matrix[0][i] * PyMath.__matrix_determinant_recursive(new_matrix)
            sign *= -1

        return new_total

    @staticmethod
    def poly_matrix_determinant(matrix):
        if len(matrix) != len(matrix[0]):
            raise ValueError("Matrix rows and columns not equal size")

        return PyMath.__poly_matrix_determinant_recursive(matrix)

    @staticmethod
    def __poly_matrix_determinant_recursive(matrix):
        if len(matrix) == 2:
            coefs1, powers1 = PyMath.multiply_poly(matrix[0][0][0], matrix[0][0][1], matrix[1][1][0], matrix[1][1][1])
            coefs2, powers2 = PyMath.multiply_poly(matrix[0][1][0], matrix[0][1][1], matrix[1][0][0], matrix[1][0][1])
            return PyMath.poly_subtract(coefs1, powers1, coefs2, powers2)

        new_totalc = [0]
        new_totalp = [0]
        sign = 1
        for i in range(len(matrix[0])):
            new_matrix = []
            if i == 0:
                for row in range(len(matrix) - 1):
                    new_matrix.append([matrix[row + 1][x + 1] for x in range(len(matrix) - 1)])
            elif i == len(matrix[0]) - 1:
                for row in range(len(matrix) - 1):
                    new_matrix.append([matrix[row + 1][x] for x in range(len(matrix) - 1)])
            else:
                for row in range(len(matrix) - 1):
                    new_row = []
                    for column in range(len(matrix[row + 1])):
                        if column != i:
                            new_row.append(matrix[row + 1][column])
                    new_matrix.append(new_row)

            new_matrix = np.array(new_matrix, dtype=object)
            mult_detc, mult_detp = PyMath.__poly_matrix_determinant_recursive(new_matrix)
            mult_detc, mult_detp = PyMath.multiply_poly(matrix[0][i][0], matrix[0][i][1], mult_detc, mult_detp)
            mult_detc, mult_detp = PyMath.multiply_poly([sign], [0], mult_detc, mult_detp)
            new_totalc, new_totalp = PyMath.poly_add(new_totalc, new_totalp, mult_detc, mult_detp)
            sign *= -1
        return new_totalc, new_totalp

    @staticmethod
    def matrix_trace(matrix):
        trace = 0
        for i in range(len(matrix)):
            trace += matrix[i][i]
        return trace

    @staticmethod
    def matrix_characteristic(matrix):
        poly_rows = np.empty([len(matrix), len(matrix)], list)
        #for x in range(len(poly_rows)):
            #for y in range(len(poly_rows[x])):
                #poly_rows[x][y] = [[], []]

        for x in range(len(matrix)):
            for y in range(len(matrix[x])):
                if x == y:
                    poly_rows[x][y] = [[matrix[x][y], -1], [0, 1]]
                else:
                    poly_rows[x][y] = [[matrix[x][y]], [0]]

        print("finding determinant")
        determc, determp = PyMath.poly_matrix_determinant(poly_rows)
        return determc, determp

    @staticmethod
    def matrix_eigen_value(matrix):
        char_c, char_p = PyMath.matrix_characteristic(matrix)
        print(PyMath.poly_to_string(char_c, char_p))
        return PyMath.poly_zeros(char_c, char_p)

    @staticmethod
    def matrix_to_string(matrix):
        result = ""
        for row in range(len(matrix)):
            result += "["
            for column in range(len(matrix[row]) - 1):
                result += str(matrix[row][column]) + ", "
            result += str(matrix[row][column + 1]) + "]\n"
        return result

    @staticmethod
    def poly_matrix_to_string(matrix):
        result = ""
        for row in range(len(matrix)):
            result += "["
            for column in range(len(matrix[row]) - 1):
                temp_pos = matrix[row][column]
                result += PyMath.poly_to_string(temp_pos[0], temp_pos[1]) + ", "
            temp_pos = matrix[row][column + 1]
            result += PyMath.poly_to_string(temp_pos[0], temp_pos[1]) + "]\n"
        return result

    @staticmethod
    def adjacency_matrix(vertices, edges):
        new_adjacency = np.zeros([len(vertices), len(vertices)], int)
        for edge in edges:
            new_adjacency[edge[0], edge[1]] = 1
            new_adjacency[edge[1], edge[0]] = 1
        return new_adjacency

    @staticmethod
    def degree_matrix(matrix):
        new_degree = np.zeros([len(matrix), len(matrix)], int)
        for i in range(len(matrix)):
            new_degree[i][i] = sum(matrix[i])
        return new_degree

    @staticmethod
    def graph_laplacian(ad_matrix, deg_matrix):
        ad_matrix = PyMath.matrix_scalar(-1, ad_matrix)
        graph_l = PyMath.add_matrices(deg_matrix, ad_matrix)
        return graph_l


# matrix = [[1, 4, 4, 6, 6, 8], [5, 3, 7, 3, 7, 3], [1, 3, 9, 0, 4, 5], [0, 2, 6, 9, 5, 9], [3, 0, 7, 8, 2, 4], [6, 6, 9, 2, 8, 7]]
# print("Eigenvalues of matrix: ")
# print(PyMath.matrix_to_string(matrix))
# print()
# print(PyMath.matrix_eigen_value(matrix))

vertices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
edges = [[5, 3], [4, 3], [4, 5], [5, 6], [5, 7], [6, 7],
         [2, 1], [2, 0], [1, 0], [0, 9], [9, 8], [0, 8], [5, 0]
         ]
adjacency = PyMath.adjacency_matrix(vertices, edges)
print(adjacency)
print()
degree = PyMath.degree_matrix(adjacency)
print(degree)
print()
graph_lap = PyMath.graph_laplacian(adjacency, degree)
print(graph_lap)
print()
eigen_graph = PyMath.matrix_eigen_value(graph_lap)
print(eigen_graph)
print()