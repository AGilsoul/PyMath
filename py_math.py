import numpy as np
import math


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

        simp_coefs = []
        simp_powers = []
        excluded_indices = []
        changes = -1
        while changes != 0:
            changes = 1
            simp_coefs = []
            simp_powers = []

            excluded_indices = []
            for p1 in range(len(new_powers)):
                if p1 not in excluded_indices:
                    duplicate_coefs = [new_coefs[p1]]
                    for p2 in range(p1 + 1, len(new_powers)):
                        if new_powers[p1] == new_powers[p2]:
                            excluded_indices.append(p2)
                            duplicate_coefs.append(new_coefs[p2])
                            changes += 1

                    simp_powers.append(new_powers[p1])
                    simp_coefs.append(sum(duplicate_coefs))

            new_powers = simp_powers
            new_coefs = simp_coefs
            changes -= 1

        return simp_coefs, simp_powers

    @staticmethod
    def poly_to_string(coefs, powers):
        result = ""
        for x in range(len(coefs)):
            if powers[x] != 0:
                if x != len(coefs) - 1:
                    result += str(coefs[x]) + "x^" + str(powers[x]) + " "
                else:
                    result += str(coefs[x]) + "x^" + str(powers[x])
            else:
                if x != len(coefs) - 1:
                    result += str(coefs[x]) + " "
                else:
                    result += str(coefs[x])
        return result

    @staticmethod
    def poly_zeros(coefs, powers):
        num_zeros = max(powers)
        known_zeros = []
        for i in range(1, 1010):
            result = PyMath.newton_method(coefs, powers, i, 0)
            if result not in known_zeros:
                known_zeros.append(result)

    @staticmethod
    def intermediate_value(coefs, powers, num_zeros):
        ranges = [[-100, 100]]
        result = PyMath.__intermediate_value_recursive(coefs, powers, num_zeros, ranges, 0)
        new_result = []
        for i in result:
            new_result.append(i[0])
            new_result.append(i[1])
            new_result.append((i[1] - i[0]) / 2)
        new_result = np.unique(np.array(new_result))
        return new_result

    @staticmethod
    def __intermediate_value_recursive(coefs, powers, num_zeros, cur_ranges, iterations):
        new_ranges = []
        for r in cur_ranges:
            temp_range1 = [r[0], r[1] - (r[1] - r[0]) / 2]
            temp_range2 = [r[1] - ((r[1] - r[0]) / 2), r[1]]
            new_ranges.append(temp_range1)
            new_ranges.append(temp_range2)

        test_ranges = []
        for r in new_ranges:
            results = PyMath.calc_polynomial(coefs, powers, r[0]), PyMath.calc_polynomial(coefs, powers, r[1])
            if results[0] * results[1] <= 0:
                test_ranges.append(r)
        if len(test_ranges) == num_zeros or len(new_ranges) >= 800:
            return test_ranges

        return PyMath.__intermediate_value_recursive(coefs, powers, num_zeros, new_ranges, iterations)

    @staticmethod
    def newton_method(coefs, powers, starting_points):
        zeros = []
        #print(starting_points)
        for i in starting_points:
            real_val = PyMath.__newton_method_recursive(coefs, powers, i, 0)
            if real_val is not None:
                zeros.append(round(real_val, 5))
            complex_val = PyMath.__newton_method_recursive(coefs, powers, complex(i, 1), 0)
            #print(complex_val)
            if complex_val is not None:
                zeros.append(complex_val)

        result = np.unique(np.array(zeros))
        th = 0.001
        new_result = np.delete(result, np.argwhere(np.ediff1d(result) <= th) + 1)
        return new_result

    @staticmethod
    def __newton_method_recursive(coefs, powers, starting_point, estimate_count):
        if estimate_count >= 200:
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

        return PyMath.__matrix_determinant_recursive(matrix, 1)

    @staticmethod
    def __matrix_determinant_recursive(matrix, sign):
        if len(matrix) == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

        new_total = 0
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
            new_total += sign * matrix[0][i] * PyMath.__matrix_determinant_recursive(new_matrix, sign)
            sign *= -1

        return new_total

    @staticmethod
    def matrix_trace(matrix):
        trace = 0
        for i in range(len(matrix)):
            trace += matrix[i][i]
        return trace

    @staticmethod
    def matrix_characteristic(matrix):
        poly_rows = np.empty([len(matrix), len(matrix)], list)

        for x in range(len(poly_rows)):
            for y in range(len(poly_rows[x])):
               poly_rows[x][y] = [[], []]

        for x in range(len(matrix)):
            for y in range(len(matrix[x])):
                if x == y:
                    poly_rows[x][y] = [[matrix[x][y], -1], [0, 1]]
                else:
                    poly_rows[x][y] = [[matrix[x][y]], [0]]

        print("[")
        for x in range(len(poly_rows)):
            print("[")
            for y in range(len(poly_rows[x])):
                if y != len(poly_rows[x]) - 1:
                    print(PyMath.poly_to_string(poly_rows[x][y][0], poly_rows[x][y][1]) + ", ")
                else:
                    print(PyMath.poly_to_string(poly_rows[x][y][0], poly_rows[x][y][1]))
            print("]")
        print("]")



matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
PyMath.matrix_characteristic(matrix)
