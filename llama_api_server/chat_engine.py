B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def dialog_to_llama_prompt(dialog):
    output = ""
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS + dialog[0]["content"] + E_SYS + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    for prompt, answer in zip(
        dialog[::2],
        dialog[1::2],
    ):
        output += (
            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
        )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    output += f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"

    return output
