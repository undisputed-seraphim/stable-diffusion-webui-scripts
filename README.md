# Stable Diffusion WebUI Scripts

Scripts written by me for use with [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

To use these scripts, simply drop them into webui root/scripts.

There may be several other repositories out there that have similar sounding names. This repository has nothing to do with them. I simply picked an unimaginative name, is all.

## Multi Prompt Search and Replace

Searches and replaces groups of keywords independently of each other. This works similarly to Prompt S/R in XY grid, but with up to 4 groups.

For example, you might want to scale (cat:1.6) downwards and (dog:0.4) upwards, to avoid confusing the model. To do this, simply enter (cat:1.6) in Group 1 prompts, and specify 1.6, 1.4, 1.2, 1.0 in Group 1 scale, and then enter (dog:0.4) in Group 2 prompts, and 0.4, 0.8, 1.2, 1.4 in Group 2 scale.

Note that each group is appended to the end of your existing prompt and the same keywords should not occur in the main prompt for consistency.

## More coming

When I think of more scripts to write.

## My favourites

[video](https://github.com/memes-forever/Stable-diffusion-webui-video)

Unfortunately it looks like the script is broken at the moment. I will try to fix it when I find the time.

# License
MIT
