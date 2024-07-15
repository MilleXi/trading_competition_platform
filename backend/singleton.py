class SingletonMeta(type):
  """
  这是一个线程安全的单例模式实现
  """
  _instances = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      instance = super().__call__(*args, **kwargs)
      cls._instances[cls] = instance
    return cls._instances[cls]
