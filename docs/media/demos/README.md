# Project 3 demo recordings

The main README will show GIF previews of our LocateAnything runs, followed by
links to the corresponding full MP4 clips. Add previews only after the actual
recordings are available.

| Scene | Final asset names | What to show |
|---|---|---|
| `chairs` | `chairs.gif`, `chairs.mp4` | Build memory, then navigate to the red and blue chairs |
| `warehouse_objects` | `warehouse-objects.gif`, `warehouse-objects.mp4` | Select forklifts, fire extinguishers, and storage racks |
| `warehouse_colors` | `warehouse-colors.gif`, `warehouse-colors.mp4` | Select objects by color, such as the green table or white chair |

Record from scene startup, or show the existing map and memory before issuing
a command. Keep the command text, selected object, and robot arrival visible.
Send one command at a time and wait for `TARGET_REACHED` or `TARGET_FAILED`.
Use IDs from the current run if demonstrating direct lookup.

Keep the original recordings outside Git. Put the finished GIF and MP4 files in
this directory. Each README caption should name the scene and command, and say
whether the clip was trimmed or sped up. Keep the command-to-arrival sequence
clear; reaching the wrong object is not a successful semantic-navigation demo.

Once the clips are ready, replace the placeholder in the
[main README](../../../README.md#demos) with the actual previews and links.
The older course GIFs in `docs/media/` are reference material, not our new demos.
