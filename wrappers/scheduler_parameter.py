##### Scheduler parameters example :
def regular_scheduler(temperature, epoch, cste = 0.999):
<<<<<<< HEAD
    if temperature < 1e-3 :
=======
    if temperature < 1e-2 :
>>>>>>> 1fcb659fe0ed06502cb9d9e599aa3f13b1965581
        return temperature
    return temperature * cste

