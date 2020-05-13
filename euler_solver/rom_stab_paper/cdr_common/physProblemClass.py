class physProblemClass:
  def __init__(self,nu_mag,sigma_mag,f_field,u_field,v_field,initialConditionFunction):
    self.nu_mag = nu_mag
    self.sigma_mag = sigma_mag
    self.f_field = f_field
    self.u_field = u_field
    self.v_field = v_field
    self.initialConditionFunction = initialConditionFunction
    
