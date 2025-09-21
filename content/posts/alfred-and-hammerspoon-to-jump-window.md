+++
title = "Jump window workflow with Alfred and Hammerspoon"
author = ["Chunwei Yan"]
date = 2025-09-21
tags = ["alfred", "hammerspoon", "productivity", "macos", "automation", "tech"]
draft = false
+++

## Introduction {#introduction}

Managing multiple windows across different applications can be challenging, especially when you have dozens of windows open. While macOS provides Command+Tab for app switching and Mission Control for window overview, sometimes you need a more direct way to jump to a specific window.

This post shows how to create a powerful window switching workflow using [Alfred](https://www.alfredapp.com/) and [Hammerspoon](https://www.hammerspoon.org/). With this setup, you can:

-   Search for any open window by title
-   See which app each window belongs to
-   Jump directly to any window with a single action
-   Use application icons for better visual recognition


## Prerequisites {#prerequisites}

Before setting up this workflow, you'll need:

1.  **Alfred Powerpack** - The paid version of Alfred that enables workflows
2.  **Hammerspoon** - A powerful automation tool for macOS
    ```bash
       # Install via Homebrew
       brew install --cask hammerspoon
    ```


## Setting up Hammerspoon {#setting-up-hammerspoon}

First, we need to add two functions to your Hammerspoon configuration. Open your `~/.hammerspoon/init.lua` file (create it if it doesn't exist) and add the following code:


### Getting window list for Alfred {#getting-window-list-for-alfred}

This function retrieves all open windows and formats them for Alfred's Script Filter:

```lua
--[[
  Generates a JSON string of all open windows for Alfred.
  Format: { "items": [ { "title": "...", "subtitle": "...", "arg": "..." } ] }
--]]
function getWindowsForAlfred()
    local windows = hs.window.allWindows()
    local alfredItems = {}

    for _, win in ipairs(windows) do
        -- Only include standard windows that are not minimized
        if win:isStandard() and not win:isMinimized() then
            local app = win:application()
            if not app then goto continue end -- Skip if the app is gone

            local appName = app:name()
            local winTitle = win:title()

            -- Get the application's path for the icon
            local appPath = app:path()

            -- Exclude windows without titles and Alfred itself
            if winTitle and #winTitle > 0 and appName ~= "Alfred" then
                table.insert(alfredItems, {
                    title = winTitle,      -- Window title shown as main text
                    subtitle = appName,    -- App name shown as subtitle
                    arg = tostring(win:id()), -- Window ID passed to action
                    uid = tostring(win:id()), -- Unique ID for Alfred
                    -- Add application icon for visual recognition
                    icon = {
                        type = "fileicon",
                        path = appPath
                    }
                })
            end
        end
        ::continue::
    end

    -- Return JSON formatted for Alfred
    return hs.json.encode({ items = alfredItems })
end
```


### Focusing a window by ID {#focusing-a-window-by-id}

This function takes a window ID and brings that window to the foreground:

```lua
--[[
  Finds a window by its ID and focuses it.
  The ID is passed as a string from the command line.
--]]
function focusWindowByID(winID)
    -- The argument from the command line is a string, so convert it to a number
    local id = tonumber(winID)
    if not id then return end

    -- Find the window by its ID
    local win = hs.window.get(id)
    if win then
        -- Focus the window (brings it to front and switches to its Space)
        win:focus()
    end
end
```

After adding both functions to your `init.lua`, reload Hammerspoon by:

-   Clicking the Hammerspoon menu bar icon and selecting "Reload Config"
-   Or pressing the reload hotkey if you've set one up


## Creating the Alfred Workflow {#creating-the-alfred-workflow}

Now let's create the Alfred workflow that will use these Hammerspoon functions.


### Step-by-step Setup {#step-by-step-setup}

1.  **Open Alfred Preferences** (⌘,) and go to the Workflows tab
2.  **Create a new workflow** by clicking the "+" button at the bottom
3.  Give it a name like "Window Switcher" and optionally add a description and icon


### Workflow Components {#workflow-components}

The workflow consists of three main components connected in sequence:

{{< figure src="/ox-hugo/2025-09-21_12-16-53_screenshot.png" >}}


#### 1. Keyword Trigger {#1-dot-keyword-trigger}

```bash
/opt/homebrew/bin/hs -c 'return getWindowsForAlfred()'
```


#### 3. Run Script Action {#3-dot-run-script-action}

Add an **Actions → Run Script** object and connect it to the Script Filter:

-   Language: `/bin/bash`
-   Script:
    ```bash
      /opt/homebrew/bin/hs -c "focusWindowByID('{query}')"
    ```
-   Configure:
    -   ☐ Escaping: All options should be unchecked

The `{query}` placeholder will be replaced with the window ID selected from the Script Filter.


## Using the Workflow {#using-the-workflow}

Once everything is set up, you can use the workflow as follows:

1.  **Trigger Alfred** with your hotkey (usually ⌘Space or ⌥Space)
2.  **Type your keyword** (e.g., `w`) followed by a space
3.  **Start typing** to search for windows by title
4.  **Select a window** using arrow keys or by continuing to type
5.  **Press Enter** to jump to that window

The workflow will:

-   Show all open windows with their titles and app names
-   Display the application icon for easy recognition
-   Filter results as you type
-   Switch to the selected window, even if it's on a different Space or minimized

The final screenshot:

{{< figure src="/ox-hugo/2025-09-21_13-09-48_screenshot.png" >}}


## Troubleshooting {#troubleshooting}

If the workflow isn't working properly, here are some common issues and solutions:


### Hammerspoon command not found {#hammerspoon-command-not-found}

If you get an error about `hs` command not found:

1.  Make sure Hammerspoon is installed and running
2.  Check the path to the `hs` command:
    ```bash
       which hs
    ```
3.  Update the paths in the Alfred workflow scripts if necessary


### No windows appearing {#no-windows-appearing}

-   Ensure Hammerspoon has accessibility permissions:
    -   System Preferences → Security &amp; Privacy → Privacy → Accessibility
    -   Make sure Hammerspoon is checked
-   Reload your Hammerspoon config
-   Check Hammerspoon console for errors (click the menu bar icon → Console)


### Window not focusing {#window-not-focusing}

-   Some apps may require additional permissions
-   Try giving both Alfred and Hammerspoon full disk access
-   Certain system windows or protected apps may not be focusable


## Enhancements and Customization {#enhancements-and-customization}

Here are some ideas to extend this workflow:


### Filter by application {#filter-by-application}

You could modify `getWindowsForAlfred()` to accept an app name parameter and only return windows from that app.


### Add window preview {#add-window-preview}

Using Hammerspoon's screenshot capabilities, you could add window thumbnails to the Alfred results.


### Keyboard shortcuts for specific apps {#keyboard-shortcuts-for-specific-apps}

Create separate workflows with different keywords for specific apps (e.g., `s` for Safari windows, `c` for Chrome).


### Recent windows {#recent-windows}

Track window focus history and sort results by most recently used.


### Window actions {#window-actions}

Instead of just focusing, add additional actions like:

-   Close window
-   Minimize/maximize
-   Move to different Space
-   Resize window


## Conclusion {#conclusion}

This Alfred + Hammerspoon workflow provides a powerful and fast way to navigate between windows. It's especially useful when you have many windows open across different Spaces and applications. The combination of Alfred's excellent search interface and Hammerspoon's window management capabilities creates a tool that's both powerful and easy to use.

The beauty of this setup is its extensibility - both Alfred and Hammerspoon are highly customizable, so you can adapt this workflow to match your specific needs and workflow preferences.
