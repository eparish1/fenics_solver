class physProblemClass:
  def __init__(self,nu_mag,sigma_mag,f_mag,u_field,v_field,initialConditionFunction):
    self.nu_mag = nu_mag
    self.sigma_mag = sigma_mag
    self.f_mag = f_mag
    self.u_field = u_field
    self.v_field = v_field
    self.initialConditionFunction = initialConditionFunction
    
