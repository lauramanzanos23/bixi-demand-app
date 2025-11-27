class MyClass:
  def __init__(self, thing, df):
    self.thing=thing
    self.df=df
  def __iter__(self):
    yield "thing", self.thing
    yield "df", self.df
