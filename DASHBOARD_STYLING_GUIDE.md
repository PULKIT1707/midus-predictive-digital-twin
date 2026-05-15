# 🎨 Dashboard Styling Guide

## Overview
The MIDUS Digital Twin Dashboard now features a **consistent, professional design system** with custom CSS styling throughout.

---

## 🎨 Design System

### **Color Palette**
```css
Primary Color:    #1f77b4 (Blue)
Secondary Color:  #ff7f0e (Orange)
Success Color:    #2ca02c (Green)
Warning Color:    #d62728 (Red)
Info Color:       #9467bd (Purple)
Background Light: #f8f9fa (Light Gray)
Text Primary:     #2c3e50 (Dark Blue-Gray)
Border Color:     #dee2e6 (Gray)
```

### **Typography**
- **H1 (Title)**: Bold (700), Blue, Bottom border
- **H2 (Sections)**: Semi-bold (600), Dark text
- **H3 (Subsections)**: Semi-bold (600), Blue
- **H4 (Components)**: Medium (500), Dark text

---

## 🧩 Component Styling

### **1. Section Badges**
- Inline badges before section headers
- Blue background with white text
- Rounded pill shape
- Example: `<span class="section-badge">Section 1</span>`

### **2. Header Banner**
- Gradient purple background (667eea → 764ba2)
- White text with project description
- Rounded corners (10px)
- Features list with bullet separators

### **3. Tabs**
- Light gray background container
- White tab buttons with hover effects
- Selected tab: Blue background with white text
- Smooth transitions

### **4. Info Buttons (ℹ️)**
- Purple circular buttons
- 32px diameter
- White text
- Positioned next to section titles

### **5. Expanders**
- Light gray background
- Purple left border (4px)
- Hover effect (darker gray)
- Bold header text

### **6. Data Tables**
- Blue header row with white text
- Alternating row colors (zebra striping)
- Hover effect (light blue)
- Rounded corners with border

### **7. Buttons**
- Primary: Blue background, white text
- Hover: Darker blue with shadow and lift effect
- Rounded corners (6px)
- Bold text (600)

### **8. Metrics**
- Large blue numbers (2rem, bold)
- Dark labels (semi-bold)
- Prominent display

### **9. Select Boxes**
- Rounded corners (6px)
- Gray border (2px)
- Blue focus border with shadow
- Smooth transitions

### **10. Charts**
- Rounded corners (8px)
- Overflow hidden for clean edges
- Consistent with overall design

---

## 📐 Layout Features

### **Spacing**
- Section dividers: 30px margin top/bottom
- Container padding: 2rem top/bottom
- Consistent gaps between elements

### **Scrollbars**
- Custom styled (10px width)
- Blue thumb with light gray track
- Rounded corners (5px)
- Hover effect (darker blue)

### **Responsive Design**
- Wide layout enabled
- Sidebar collapsed by default
- Flexible columns for different screen sizes

---

## 🎯 Section Structure

### **Quick Start**
- Badge: "Quick Start"
- Icon: 🚀
- Title: "Guided Demo (one-click)"

### **Section 1**
- Badge: "Section 1"
- Icon: 📁
- Title: "Load a .sav file + preview metadata"

### **Section 2**
- Badge: "Section 2"
- Icon: 📂
- Title: "Browse generated artifacts"

### **Section 3**
- Badge: "Section 3"
- Icon: 📊
- Title: "Descriptive Statistics & Demographics"

### **Section 4**
- Badge: "Section 4"
- Icon: 🤖
- Title: "Predictive Twin (model + what-if simulation)"

---

## 📊 Footer Design

### **Features**
- Light gray background
- Blue top border (3px)
- Centered content
- Three stat cards:
  - 📊 Sample Size: N = 2,735
  - 🎯 Model Performance: R² = 0.575
  - ⚖️ Model Fairness: Unbiased ✓

### **Tech Stack Display**
- Listed with bullet separators
- Gray text
- Smaller font size

### **Copyright**
- Bottom of footer
- Small gray text
- Year and project name

---

## 🎨 Visual Hierarchy

### **Level 1: Page Title**
- Large, bold, blue
- Bottom border
- Icon included

### **Level 2: Section Headers**
- Badge + Icon + Title
- Clear separation with dividers
- Consistent spacing

### **Level 3: Subsections**
- Blue text
- Medium weight
- Info buttons aligned right

### **Level 4: Components**
- Dark text
- Smaller headers
- Inline info popovers

---

## 💡 Best Practices

### **Consistency**
✅ All sections follow same pattern (Badge → Icon → Title)
✅ All info buttons use same purple color
✅ All tables have same header styling
✅ All buttons have same hover effects

### **Accessibility**
✅ High contrast text colors
✅ Clear visual hierarchy
✅ Readable font sizes
✅ Hover states for interactive elements

### **User Experience**
✅ Smooth transitions and animations
✅ Clear section boundaries
✅ Informative tooltips and popovers
✅ Responsive layout

### **Professional Polish**
✅ Gradient header banner
✅ Custom scrollbars
✅ Rounded corners throughout
✅ Consistent spacing and alignment

---

## 🔧 Customization

### **To Change Primary Color:**
Update the `--primary-color` variable in the CSS:
```css
--primary-color: #YOUR_COLOR;
```

### **To Adjust Spacing:**
Modify the `.block-container` padding:
```css
.block-container {
    padding-top: YOUR_VALUE !important;
    padding-bottom: YOUR_VALUE !important;
}
```

### **To Change Tab Style:**
Update `.stTabs` selectors in the CSS section.

---

## 📱 Responsive Behavior

- **Desktop**: Full wide layout with all features
- **Tablet**: Columns stack appropriately
- **Mobile**: Single column layout (Streamlit default)

---

## ✨ Special Features

### **Gradient Header**
- Eye-catching purple gradient
- White text for contrast
- Project description and features

### **Section Badges**
- Quick visual navigation
- Numbered sections
- Consistent placement

### **Info Ecosystem**
- Section-level expanders (large explanations)
- Component-level popovers (quick help)
- Consistent ℹ️ icon usage

### **Footer Stats**
- Key metrics at a glance
- White cards on gray background
- Professional presentation

---

## 🎓 Usage in Presentations

### **For Demos:**
1. Point out the clean, professional design
2. Show the consistent color scheme
3. Demonstrate the info buttons
4. Highlight the footer stats

### **For Stakeholders:**
1. Emphasize the polished appearance
2. Show the clear section structure
3. Demonstrate ease of navigation
4. Highlight the comprehensive documentation

### **For Technical Reviewers:**
1. Show the custom CSS implementation
2. Demonstrate responsive design
3. Highlight accessibility features
4. Show the consistent component library

---

## 📝 Maintenance Notes

- All CSS is contained in the main `demo_app.py` file
- Uses Streamlit's `unsafe_allow_html=True` for custom styling
- Compatible with Streamlit version 1.x
- No external CSS files required
- Easy to update and maintain

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Dark mode toggle
- [ ] Customizable color themes
- [ ] Export styling to separate CSS file
- [ ] Additional animation effects
- [ ] Mobile-optimized layouts
- [ ] Print-friendly styles

---

**Last Updated:** May 1, 2026
**Version:** 2.0
**Status:** Production Ready ✓
