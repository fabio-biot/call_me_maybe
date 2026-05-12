from pydantic import BaseModel


class Parameter(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: dict[str, str]


class PromptInput(BaseModel):
    prompt: str


class FunctionCall(BaseModel):
    name: str
    prompt: str
    parameters: dict[str, str]

