class constraintProps:
	def __init__(self, mutable=True, step_direction=0):
		self._mutable = mutable
		self._step_direction = step_direction

	@property
	def mutable(self):
		return self._mutable

	@mutable.setter
	def mutable(self, v):
		self._mutable = v

	@property
	def step_direction(self):
		return self._step_direction

	@step_direction.setter
	def step_direction(self, v):
		self._step_direction = v

	def set_mutability(self, m):
		self._mutable = m

	def set_direction(self, s):
		self._step_direction = s
	

class feasibility_consts:
    def __init__(self, feature_columns):
        self._feature_columns = feature_columns
        self._feasibility_set = {}
        self._feature_columns_to_check = self._feature_columns
        self._initialize_constraints()

    def _initialize_constraints(self):
        for i, feat in enumerate(self._feature_columns):
            self._feasibility_set[i] = constraintProps()

    def set_constraint(self, feat_pattern, mutability=True, step_direction=0, exact_match=False):
        """
        # Set constraints for a feature or a group of features

        # Parameters:
        ------------
        # feat_pattern: (str)
        - The feature or a group of features to set constraints for
        # mutability: (bool)
        - Whether the feature is mutable or not
        # step_direction: (int)
        - The direction of the step. 1 for positive, -1 for negative
        # exact_match: (bool)
        - If True, the feature name should match exactly.
        """
        for i, feat in enumerate(self._feature_columns):
            if (exact_match and feat == feat_pattern) or (not exact_match and feat_pattern in feat):
                self._feasibility_set[i] = constraintProps(mutability, step_direction)

    def set_feature_columns_to_check(self, exclude_columns=[]):
          self._feature_columns_to_check = list(set(self._feature_columns) - set(exclude_columns))

    def check_constraints(self, source, dest):
        """
        # Check if the constraints are satisfied

        # Parameters:
        ------------
        # source: (list)
        - The source feature values
        # dest: (list)
        - The destination feature values

        # Returns:
        ------------
        - bool: True if the constraints are satisfied, False otherwise
        """
        delta = dest - source
        for i, _ in enumerate(self._feature_columns_to_check):
            if ((delta[i] != 0) and (self._feasibility_set[i]._mutable is False)):
                return False
            if (delta[i] * self._feasibility_set[i]._step_direction < 0):
                return False
        return True

    def check_constraints_german_credit(self, source, dest):
        """
        # Check if the constraints are satisfied for the German Credit dataset

        # Parameters:
        ------------
        # source: (list)
        - The source feature values
        # dest: (list)
        - The destination feature values

        # Returns:
        ------------
        - bool: True if the constraints are satisfied, False otherwise
        """
        delta = dest - source
        credit_history_high = ['Credit-History_3', 'Credit-History_4']
        credit_history_low = ['Credit-History_0', 'Credit-History_1', 'Credit-History_2']
        high_indices = [list(self._feature_columns).index(ch) for ch in credit_history_high if ch in self._feature_columns]
        low_indices = [list(self._feature_columns).index(ch) for ch in credit_history_low if ch in self._feature_columns]
        for idx in high_indices:
            if (source[idx] == 1 and dest[idx] == 0):
                # Dont allow connections to higher indices
                for idx2 in high_indices:
                    if idx2 > idx and dest[idx2] == 1:
                        return False
        for idx in low_indices:
            if (source[idx] == 1 and dest[idx] == 0):
                if any((dest[idx2] == 1) for idx2 in high_indices):
                    return False
        # Check all other features constraints as the previous method
        for i, feat in enumerate(self._feature_columns):
            if feat not in credit_history_high + credit_history_low:
                if ((delta[i] != 0) and (self._feasibility_set[i]._mutable is False)):
                    return False
                if (delta[i] * self._feasibility_set[i]._step_direction < 0):
                    return False
        return True