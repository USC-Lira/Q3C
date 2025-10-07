import numpy as np

from large_rl.commons.plot_embed import plot_embedding


def visualise_embed(env, file_name: str,
                    if_plot_proto_act: bool = True,
                    if_plot_task_repre: bool = False,
                    proto_action: np.ndarray = None,
                    user_embed: np.ndarray = None,
                    act_embed: np.ndarray = None,
                    title: str = None,
                    save_dir: str = "./images/"):
    if env.args["env_name"].lower() == "mine":
        actions = env.act_embedding
    elif env.args["env_name"].startswith("recsim"):
        nested_actions = env.slate_to_doc_cls(slates=np.arange(env.act_embedding.shape[0]))
        actions = [_a for a in nested_actions for _a in a]
    else:
        raise ValueError
    label_dict = {"labels": list()}

    for action in actions:
        label_dict["labels"].append(f"{action.cluster_id}")

    if proto_action is not None:
        for _a in proto_action:
            # label_dict["labels"].append(f"q_{np.argmax(_a)}")
            label_dict["labels"].append("a")

    if user_embed is not None:
        label_dict["labels"].append("s")

    task_embed, action_embed = np.asarray(env.task_embedding), np.asarray(env.act_embedding)

    label_dict["desc"] = {i: i for i in set(label_dict["labels"])}
    label_dict["labels"] = np.asarray(label_dict["labels"])

    img_path = list()
    if act_embed is not None:
        if proto_action is not None: act_embed = np.vstack([act_embed, proto_action])
        if user_embed is not None: act_embed = np.vstack([act_embed, user_embed])
        plot_embedding(embedding=act_embed, label_dict=label_dict, save_dir=save_dir, file_name=file_name, title=title)
        img_path += [file_name]
    if if_plot_proto_act:
        if proto_action is not None: action_embed = np.vstack([action_embed, proto_action])
        if user_embed is not None: action_embed = np.vstack([action_embed, user_embed])
        plot_embedding(embedding=action_embed, label_dict=label_dict, save_dir=save_dir, title=title,
                       file_name=f"{file_name}_action_embed")
        img_path += [f"{file_name}_action_embed"]
    if if_plot_task_repre:
        plot_embedding(embedding=task_embed, label_dict=label_dict, save_dir=save_dir, file_name="task", title=title)
        img_path += ["task"]
    return img_path
