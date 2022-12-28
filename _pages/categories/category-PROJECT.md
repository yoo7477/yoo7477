---
title: "PROJECT"
layout: archive
permalink: categories/PROJECT
author_profile: true
sidebar_main: true
---


{% assign posts = site.categories.PROJECT %}
{% for post in posts %} 
  {% include archive-single.html type=page.entries_layout %} 
{% endfor %}
